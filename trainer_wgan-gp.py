import torch
import json
from models.Encoder import baseEncoder, baseEncoderv2, harchyEncoder
from models.Decoder import baseDecoder, baseDecoderv2,  baseDecoderv3

from models.Discriminator import SNDiscriminator, baseDiscriminator, noCon_Discriminator, PatchDiscriminator, \
    ResDiscriminator, PDiscriminator
from models.tools import cal_gradient_penalty, init_weights
from torch import nn
from utils.MIMICDataSet import MIMICDataset2
from utils.OpeniDataSet import OpeniDataset2
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from utils import get_time, matplotlib_imshow, deNorm, Rescale, ToTensor, Equalize
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR


class Trainer:
    def __init__(self):
        self.cfg_json = "config/MIMIC_wgan.json"
        self.cfg = self.pare_cfg(self.cfg_json)
        self.exp_name = self.cfg["EXPER_NAME"]
        self.max_epoch = self.cfg["MAX_EPOCH"]


        self.encoder_checkpoint = self.cfg["CHECKPOINT_ENCODER"]
        self.decoder_checkpoint = self.cfg["CHECKPOINT_DECODER"]

        self.D_checkpoint = self.cfg["CHECKPOINT_D"]
        self.check_create_checkpoint()

        self.encoder_resume = self.cfg["RESUME_ENCODER"]
        self.decoder_resume_F = self.cfg["RESUME_DECODER_F"]
        self.decoder_resume_L = self.cfg["RESUME_DECODER_L"]
        self.D_resume = self.cfg["RESUME_D"]

        self.train_csv = 'openi_report_train.csv'
        self.val_csv = 'openi_report_val.csv'
        self.test_csv = 'openi_report_test.csv'

        self.img_csv = self.cfg["IMG_CSV"]
        self.data_root = self.cfg["DATA_ROOT"]
        self.batch_size = self.cfg["BATCH_SIZE"]
        self.image_size = tuple(self.cfg["IMAGE_SIZE"])
        self.name = self.cfg["EXPER_NAME"]
        self.pix_loss_ratio = self.cfg["PIXEL_LOSS_RATIO"]
        self.adv_loss_ratio = self.cfg["ADV_LOSS_RATIO"]
        self.checkpoint_epoch = self.cfg["CHECKPOINT_EPOCH"]

        self.beta1 = self.cfg["beta1"]
        self.word_dict = self.cfg["DICTIONARY"]
        self.writer = SummaryWriter(os.path.join("runs", self.exp_name))
        self.ENCODERS = {
            "baseENCODER": baseEncoder,
            "baseENCODERv2": baseEncoderv2,
            "harchyENCODER": harchyEncoder
        }
        self.DECODERS = {
            "baseDECODER": baseDecoder,
            "baseDECODERv2": baseDecoderv2,
            "baseDECODERv3": baseDecoderv3
        }

        self.DISCRIMINATOR = {
            "baseDISCRIMINATOR": baseDiscriminator,
            "noconDISCRIMINATOR": noCon_Discriminator,
            "Patch": PatchDiscriminator,
            "SNDiscriminator": SNDiscriminator,
            "ResDISCRIMINATOR": ResDiscriminator,
            "PDISCRIMINATOR": PDiscriminator
        }
        self.dataset = {
            "OPENI": OpeniDataset2,
            "MIMIC-CXR": MIMICDataset2
        }

        ##################################################
        ################# Dataset Setup ##################
        ##################################################
        self.trainset = self.dataset[self.cfg["DATASET"]](csv_txt=self.train_csv,
                                                             csv_img=self.img_csv,
                                                             root=self.data_root,
                                                             word_dict=self.word_dict,
                                                             transform=transforms.Compose([
                                                                 Rescale(self.image_size),
                                                                 Equalize(),
                                                                 ToTensor()
                                                             ]))

        self.valset = self.dataset[self.cfg["DATASET"]](csv_txt=self.val_csv,
                                                             csv_img=self.img_csv,
                                                             root=self.data_root,
                                                             word_dict=self.word_dict,
                                                             transform=transforms.Compose([
                                                                 Rescale(self.image_size),
                                                                 Equalize(),
                                                                 ToTensor()
                                                             ]))

        self.sia_dataset = self.dataset[self.cfg["DATASET"]][1](csv_txt=self.train_csv,
                                                csv_img=self.img_csv,
                                                root=self.data_root,
                                                transform=transforms.Compose([
                                                         Rescale(self.image_size),
                                                         Equalize(),
                                                         ToTensor()
                                                        ]))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        s_gpus = self.cfg["GPU_ID"].split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)

        #########################################
        ############ Loss Function ##############
        #########################################
        content_losses = {"L2": nn.MSELoss(),
                          "L1": nn.L1Loss()}
        self.G_criterion = content_losses[self.cfg["CONTENT_LOSS"]].to(self.device)

        #########################################
        ############ Network Init ###############
        #########################################



        self.base_size = self.image_size[0]
        self.base_ratio = int(np.log2(self.base_size))


        self.define_nets()
        if self.num_gpus > 1:
            self.encoder = nn.DataParallel(self.encoder, device_ids=self.gpus)
            self.decoder_L = nn.DataParallel(self.decoder_L, device_ids=self.gpus)
            self.decoder_F = nn.DataParallel(self.decoder_F, device_ids=self.gpus)

    def define_nets(self):
        self.encoder = self.ENCODERS[self.cfg["ENCODER"]](vocab_size=self.t2i_dataset.vocab_size,
                                                          embed_size=self.cfg["E_EMBED_SIZE"],
                                                          hidden_size=self.cfg["E_HIDEN_SIZE"],
                                                          max_len=[self.t2i_dataset.max_len_finding,
                                                                   self.t2i_dataset.max_len_impression],
                                                          unit=self.cfg["RNN_CELL"],
                                                          feature_base_dim=self.cfg["D_CHANNEL_SIZE"]
                                                          ).to(self.device)
        # self.encoder.apply(init_weights)

        self.decoder_F = self.DECODERS[self.cfg["DECODER"]](input_dim=self.cfg["D_CHANNEL_SIZE"],
                                                           feature_base_dim=self.cfg["D_CHANNEL_SIZE"],
                                                           uprate=self.base_ratio).to(self.device)





        self.decoder_L = self.DECODERS[self.cfg["DECODER"]](input_dim=self.cfg["D_CHANNEL_SIZE"],
                                                           feature_base_dim=self.cfg["D_CHANNEL_SIZE"],
                                                           uprate=self.base_ratio).to(self.device)


    def define_dataloader(self):

        self.train_dataloader = DataLoader(self.trainset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=8,
                                           drop_last=True)
        self.val_dataloader = DataLoader(self.valset,
                                         batch_size=self.batch_size,
                                         shuffle=True,
                                         num_workers=8,
                                         drop_last=True)

    def define_opt(self, only_G=False):
        '''Define optimizer'''
        self.G_optimizer = torch.optim.Adam([{'params': self.encoder.parameters()}] +
                                            [{'params': self.decoder_F.parameters()}] +
                                            [{'params': self.decoder_L.parameters()}],
                                            lr=self.cfg["G_LR"], betas=(self.beta1, 0.999))
        self.G_lr_scheduler = MultiStepLR(self.G_optimizer, milestones=self.cfg["LR_DECAY_EPOCH"], gamma=0.2)

        self.D_optimizer = torch.optim.Adam([{'params': self.D_F.parameters()}] +
                                            [{'params': self.D_L.parameters()}]
                                            , lr=self.cfg["D_LR"], betas=(self.beta1, 0.999))

        self.D_lr_scheduler = MultiStepLR(self.D_optimizer, milestones=self.cfg["LR_DECAY_EPOCH"], gamma=0.2)

    def check_create_checkpoint(self):
        '''Check for the checkpoint path exists or not
        If not exist, create folder'''
        if os.path.exists(self.encoder_checkpoint) == False:
            os.makedirs(self.encoder_checkpoint)
        if os.path.exists(self.decoder_checkpoint) == False:
            os.makedirs(self.decoder_checkpoint)
        if os.path.exists(self.D_checkpoint) == False:
            os.makedirs(self.D_checkpoint)

    def load_model(self):
        self.encoder.load_state_dict(torch.load(self.encoder_resume))

        self.decoder_F.load_state_dict(torch.load(self.decoder_resume_F))
        self.decoder_L.load_state_dict(torch.load(self.decoder_resume_L))

    def define_D(self):
        '''Initialize a series of Discriminator'''

        dr = self.base_ratio - 2
        self.D_F = self.DISCRIMINATOR[self.cfg["DISCRIMINATOR"]](base_feature=self.cfg["DIS_CHANNEL_SIZE"],
                                                                 txt_input_dim=self.cfg["D_CHANNEL_SIZE"],
                                                                 down_rate=dr).to(self.device)

        self.D_L = self.DISCRIMINATOR[self.cfg["DISCRIMINATOR"]](base_feature=self.cfg["DIS_CHANNEL_SIZE"],
                                                                 txt_input_dim=self.cfg["D_CHANNEL_SIZE"],
                                                                 down_rate=dr).to(self.device)
        # self.D.apply(init_weights)
        if self.num_gpus > 1:
            self.D_F = nn.DataParallel(self.D_F, device_ids=self.gpus)
            self.D_L = nn.DataParallel(self.D_L, device_ids=self.gpus)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def Loss_on_layer(self, image, finding, impression, decoder):
        '''
        Pretrain genertaor with batch
        :image image batch
        :text text batch
        '''

        txt_emded, hidden = self.encoder(finding, impression)


        self.G_optimizer.zero_grad()
        pre_image = decoder(txt_emded)
        loss = self.G_criterion(pre_image.float(), image.float())
        loss.backward()
        self.G_optimizer.step()
        return loss, pre_image, image

    def Loss_on_layer_GAN(self, image, finding, impression, decoder, D):
        '''
        Pretrain genertaor with batch
        :image image batch
        :text text batch
        '''


        txt_emded, hidden = self.encoder(finding, impression)
        pre_image = decoder(txt_emded)

        # Train Discriminator

        self.D_optimizer.zero_grad()
        pre_fake = D(pre_image, txt_emded)
        pre_real = D(image, txt_emded)
        gradient_penalty, gradients = cal_gradient_penalty(D, image, pre_image, txt_emded, "cuda")

        D_loss = pre_fake.mean() - pre_real.mean() + gradient_penalty
        D_loss.backward(retain_graph=True)

        self.D_optimizer.step()

        # Train Generator

        self.G_optimizer.zero_grad()

        pre_fake = D(pre_image, txt_emded)

        adv_loss = - self.adv_loss_ratio * pre_fake.mean()
        adv_loss.backward(retain_graph=True)

        content_loss = self.pix_loss_ratio * self.G_criterion(pre_image.float(),
                                                              image.float())
        content_loss.backward(retain_graph=True)

        G_loss = content_loss + adv_loss

        self.G_optimizer.step()

        return D_loss, G_loss, pre_image, image

    def train_layer(self):
        DISP_FREQ = 10
        for epoch in range(20):
            print('Generator Epoch {}'.format(epoch))
            self.encoder.train()
            self.decoder_F.train()
            self.decoder_L.train()
            for idx, batch in enumerate(tqdm(self.train_dataloader)):
                finding = batch['finding'].to(self.device)
                impression = batch['impression'].to(self.device)
                image_f = batch['image_F'].to(self.device)
                image_l = batch['image_L'].to(self.device)

                loss_f, pre_image_f, r_image_f = self.Loss_on_layer(image_f, finding, impression,
                                                                    self.decoder_F)
                loss_l, pre_image_l, r_image_l = self.Loss_on_layer(image_l, finding, impression,
                                                                    self.decoder_L)

                # print('Loss: {:.4f}'.format(loss.item()))
                if ((idx + 1) % DISP_FREQ == 0) and idx != 0:
                    self.writer.add_scalar('Train_front loss',
                                           loss_f.item(),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_scalar('Train_lateral loss',
                                           loss_l.item(),
                                           epoch * len(self.train_dataloader) + idx)

                    # write to tensorboard
                    self.writer.add_images("Train_front_Original",
                                           deNorm(r_image_f),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("Train_front_Predicted",
                                           deNorm(pre_image_f),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("Train_lateral_Original",
                                           deNorm(r_image_l),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("Train_lateral_Predicted",
                                           deNorm(pre_image_l),
                                           epoch * len(self.train_dataloader) + idx)

            self.G_lr_scheduler.step(epoch)

    def train_GAN_layer(self):
        DISP_FREQ = 10
        self.encoder.train()
        self.decoder_F.train()
        self.decoder_L.train()
        self.D_F.train()
        self.D_L.train()
        for epoch in range(self.max_epoch):
            print('GAN Epoch {}'.format(epoch))

            for idx, batch in enumerate(tqdm(self.train_dataloader)):
                finding = batch['finding'].to(self.device)
                impression = batch['impression'].to(self.device)
                image_f = batch['image_F'].to(self.device)
                image_l = batch['image_L'].to(self.device)

                D_loss_f, G_loss_f, pre_image_f, image_f = self.Loss_on_layer_GAN(image_f, finding, impression,self.decoder_F, self.D_F)
                D_loss_l, G_loss_l, pre_image_l, image_l = self.Loss_on_layer_GAN(image_l, finding, impression,self.decoder_L, self.D_L)

                if ((idx + 1) % DISP_FREQ == 0) and idx != 0:
                    # ...log the running loss
                    # self.writer.add_scalar("Train_{}_SSIM".format(layer_id), ssim.ssim(r_image, pre_image).item(),
                    #                        epoch * len(self.train_dataloader) + idx)
                    self.writer.add_scalar('GAN_G_train_Layer_front_loss',
                                           G_loss_f.item(),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_scalar('GAN_D_train_Layer_front_loss',
                                           D_loss_f.item(),
                                           epoch * len(self.train_dataloader) + idx)

                    self.writer.add_scalar('GAN_G_train_Layer_lateral_loss',
                                           G_loss_l.item(),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_scalar('GAN_D_train_Layer_lateral_loss',
                                           D_loss_l.item(),
                                           epoch * len(self.train_dataloader) + idx)

                    # write to tensorboard
                    self.writer.add_images("GAN_Train_Original_front",
                                           deNorm(image_f),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("GAN_Train_Predicted_front",
                                           deNorm(pre_image_f),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("GAN_Train_Original_lateral",
                                           deNorm(image_l),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("GAN_Train_Predicted_lateral",
                                           deNorm(pre_image_l),
                                           epoch * len(self.train_dataloader) + idx)
            self.G_lr_scheduler.step(epoch)
            self.D_lr_scheduler.step(epoch)
            if epoch%10==0 and epoch!=0:
                torch.save(self.encoder.state_dict(), os.path.join(self.encoder_checkpoint,
                                                                   "Encoder_{}_epoch_{}_checkpoint.pth".format(
                                                                       self.cfg["ENCODER"],
                                                                       epoch)))
                torch.save(self.D_F.state_dict(), os.path.join(self.D_checkpoint,
                                                               "D_{}_F_epoch_{}_checkpoint.pth".format(
                                                                   self.cfg["DISCRIMINATOR"],epoch)))
                torch.save(self.D_L.state_dict(), os.path.join(self.D_checkpoint,
                                                               "D_{}_L_epoch_{}_checkpoint.pth".format(
                                                                   self.cfg["DISCRIMINATOR"], epoch)))

                torch.save(self.decoder_F.state_dict(), os.path.join(self.decoder_checkpoint,
                                                                     "Decoder_{}_F_epoch_{}_checkpoint.pth".format(
                                                                         self.cfg["DECODER"], epoch)))

                torch.save(self.decoder_L.state_dict(), os.path.join(self.decoder_checkpoint,
                                                                     "Decoder_{}_L_epoch_{}_checkpoint.pth".format(
                                                                         self.cfg["DECODER"], epoch)))

    def train(self):

        # self.load_model()


        self.define_D()
        self.define_opt()
        self.define_dataloader()

        #########################################################
        ############### Train Generator by layer ################
        #########################################################
        print("Star training on Decoder")
        self.train_layer()
        #########################################################
        ################## Train GAN by layer ###################
        #########################################################
        print("Star training GAN")
        self.train_GAN_layer()


    def pare_cfg(self, cfg_json):
        with open(cfg_json) as f:
            cfg = f.read()
            print(cfg)
            print("Config Loaded")
        return json.loads(cfg)


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()
