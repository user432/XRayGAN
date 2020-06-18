import torch
import json
from models.StackGAN import *
from models.Encoder import baseEncoder,baseEncoderv2,harchyEncoder
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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class Trainer:
    def __init__(self):
        self.cfg_json = "config/MIMIC_StackGAN.json"
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
        self.D_resume_F,self.D_resume_L = self.cfg["RESUME_D"]
        self.train_csv = self.cfg["TRAIN_CSV"]
        self.val_csv = self.cfg["VAL_CSV"]
        self.test_csv = self.cfg["TEST_CSV"]
        self.text_csv = self.cfg["TEXT_CSV"]
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
        self.dataset = {
            "OPENI": OpeniDataset2,
            "MIMIC-CXR": MIMICDataset2
        }

        ##################################################
        ################# Dataset Setup ##################
        ##################################################
        self.t2i_dataset = self.dataset[self.cfg["DATASET"]](csv_txt=self.text_csv,
                                                             csv_img=self.img_csv,
                                                             root=self.data_root,
                                                             word_dict=self.word_dict,
                                                             transform=transforms.Compose([
                                                                 Rescale(self.image_size),
                                                                 Equalize(),
                                                                 ToTensor()
                                                             ]))
        self.trainset = self.dataset[self.cfg["DATASET"]][0](csv_txt=self.train_csv,
                                                             csv_img=self.img_csv,
                                                             root=self.data_root,
                                                             word_dict=self.word_dict,
                                                             transform=transforms.Compose([
                                                                 Rescale(self.image_size),
                                                                 Equalize(),
                                                                 ToTensor()
                                                             ]))

        self.valset = self.dataset[self.cfg["DATASET"]][0](csv_txt=self.val_csv,
                                                             csv_img=self.img_csv,
                                                             root=self.data_root,
                                                             word_dict=self.word_dict,
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


        self.decoder_L,self.D_L= self.define_nets()
        self.decoder_F,self.D_F= self.define_nets()
        self.encoder = harchyEncoder(vocab_size=self.t2i_dataset.vocab_size,
                                     embed_size=self.cfg["E_EMBED_SIZE"],
                                     hidden_size=self.cfg["E_HIDEN_SIZE"],
                                     max_len=[self.t2i_dataset.max_len_finding,
                                              self.t2i_dataset.max_len_impression],
                                     unit=self.cfg["RNN_CELL"],
                                     feature_base_dim=self.cfg["D_CHANNEL_SIZE"]
                                     ).to(self.device)

        self.encoder = nn.DataParallel(self.encoder, device_ids=self.gpus)
    def define_nets(self):

        netG = G_NET().to(self.device)
        # netG.apply(weights_init)
        netG = nn.DataParallel(netG, device_ids=self.gpus).to(self.device)
        print(netG)

        netsD = []

        netsD.append(D_NET64().to(self.device))

        netsD.append(D_NET128().to(self.device))

        netsD.append(D_NET256().to(self.device))

        for i in range(len(netsD)):
            # netsD[i].apply(weights_init)
            netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=self.gpus).to(self.device)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        return netG,netsD


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

        self.D_optimizer = torch.optim.Adam([{'params': D.parameters()} for D in self.D_F] +
                                            [{'params': D.parameters()}for D in self.D_L]
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
        print("Model Loading....................")
        self.encoder.load_state_dict(torch.load(self.encoder_resume))
        self.decoder_F.load_state_dict(torch.load(self.decoder_resume_F))
        self.decoder_L.load_state_dict(torch.load(self.decoder_resume_L))
        for i in range(3):
            self.D_L[i].load_state_dict(torch.load(self.D_resume_L[i]))
            self.D_F[i].load_state_dict(torch.load(self.D_resume_F[i]))

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def Loss_on_layer(self, image, finding, impression, decoder):
        '''
        Pretrain genertaor with batch
        :image image batch
        :text text batch
        '''
        image1 = F.interpolate(image, size=(64,64))
        image2 = F.interpolate(image, size=(128,128))
        image3 = F.interpolate(image, size=(256,256))
        real_img = [image1, image2, image3]
        txt_emded, hidden = self.encoder(finding, impression)

        self.G_optimizer.zero_grad()
        pre_image,mu, logvar = decoder(txt_emded)
        total_loss = 0
        for i in range(3):
            loss = self.G_criterion(pre_image[i].float(), real_img[i].float())

            loss.backward(retain_graph=True)
            total_loss+=loss
        self.G_optimizer.step()
        return total_loss, pre_image[-1], real_img[-1]

    def Loss_on_layer_GAN(self, image, finding, impression, decoder, D):
        '''
        Pretrain genertaor with batch
        :image image batch
        :text text batch
        '''

        image1 = F.interpolate(image, size=(64, 64))
        image2 = F.interpolate(image, size=(128, 128))
        image3 = F.interpolate(image, size=(256, 256))
        real_img = [image1,image2,image3]
        txt_emded, hidden = self.encoder(finding, impression)
        pre_image, mu, logvar = decoder(txt_emded)

        # Train Discriminator

        self.D_optimizer.zero_grad()
        total_D_loss = 0

        for i in range(3):
            pre_fake = D[i](pre_image[i], mu.detach())
            pre_real = D[i](real_img[i], mu.detach())
            gradient_penalty, gradients = cal_gradient_penalty(D[i], real_img[i], pre_image[i], mu.detach(), "cuda")

            D_loss = pre_fake.mean() - pre_real.mean() + gradient_penalty
            total_D_loss += D_loss
            D_loss.backward(retain_graph=True)

        self.D_optimizer.step()

        # Train Generator

        self.G_optimizer.zero_grad()
        G_loss = 0
        for i in range(3):
            pre_fake = D[i](pre_image[i], mu.detach())

            adv_loss = - self.adv_loss_ratio * pre_fake.mean()
            adv_loss.backward(retain_graph=True)

            content_loss = self.pix_loss_ratio * self.G_criterion(pre_image[i].float(),
                                                                  real_img[i].float())
            content_loss.backward(retain_graph=True)

            G_loss += content_loss
            G_loss += adv_loss

        self.G_optimizer.step()

        return total_D_loss, G_loss, pre_image[-1], real_img[-1]

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
    def set_D(self,Ds, training = True):
        if training:
            for D in Ds:
                D.train()
        else:
            for D in Ds:
                D.eval()

    def train_GAN_layer(self):
        DISP_FREQ = 10
        self.encoder.train()
        self.decoder_F.train()
        self.decoder_L.train()
        self.set_D(self.D_F)
        self.set_D(self.D_L)
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
                for i in range(3):
                    torch.save(self.D_F[i].state_dict(), os.path.join(self.D_checkpoint,
                                                                   "D_{}_F_epoch_{}_layer_{}_checkpoint.pth".format(
                                                                       self.cfg["DISCRIMINATOR"],epoch,i)))
                    torch.save(self.D_L[i].state_dict(), os.path.join(self.D_checkpoint,
                                                                   "D_{}_L_epoch_{}_layer_{}_checkpoint.pth".format(
                                                                       self.cfg["DISCRIMINATOR"], epoch,i)))

                torch.save(self.decoder_F.state_dict(), os.path.join(self.decoder_checkpoint,
                                                                     "Decoder_{}_F_epoch_{}_checkpoint.pth".format(
                                                                         self.cfg["DECODER"], epoch)))

                torch.save(self.decoder_L.state_dict(), os.path.join(self.decoder_checkpoint,
                                                                     "Decoder_{}_L_epoch_{}_checkpoint.pth".format(
                                                                         self.cfg["DECODER"], epoch)))

    def train(self):

        # self.load_model()

        self.define_opt()
        self.define_dataloader()

        #########################################################
        ############### Train Generator by layer ################
        #########################################################
        print("Start training on Decoder")
        # self.train_layer()
        #########################################################
        ################## Train GAN by layer ###################
        #########################################################
        print("Start training GAN")
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
