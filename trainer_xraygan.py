import torch
import json
from models.Encoder import HAttnEncoder,HAttnEncoderv2,HAttnEncoderv3,HAttnEncoderv4,plainEncoder,harchyEncoderv2, harchyEncoderv3
from models.Decoder import baseDecoder,PDecoderv2,baseDecoderv2,PDecoder,baseDecoderv3,MultiscaleDecoder,PDecoderv3
from models.Discriminator import SNDiscriminator, baseDiscriminator, noCon_Discriminator, PatchDiscriminator, ResDiscriminator, PDiscriminator
from models.tools import cal_gradient_penalty,init_weights
from torch import nn
from torch.nn.modules.distance import PairwiseDistance
from utils.MIMICDataSet import MIMICDataset2_Hiachy, MIMICDataset_Siamese
from utils.OpeniDataSet import OpeniDataset2_Hiachy,OpeniDataset_Siamese,OpeniDataset2
from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from utils import get_time,matplotlib_imshow,deNorm, Rescale,ToTensor,Equalize
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.Siamese import EmbeddingNet,Classifinet
import os
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR,StepLR

class Trainer:
    def __init__(self):
        self.cfg_json = "config/MIMIC_XrayGAN.json"
        self.cfg = self.pare_cfg(self.cfg_json)
        self.exp_name = self.cfg["EXPER_NAME"]
        self.max_epoch = self.cfg["MAX_EPOCH"]
        self.S_max_epoch = self.cfg["SIAMESE_EPOCH"]

        self.encoder_checkpoint = self.cfg["CHECKPOINT_ENCODER"]
        self.decoder_checkpoint = self.cfg["CHECKPOINT_DECODER"]

        self.D_checkpoint = self.cfg["CHECKPOINT_D"]
        self.check_create_checkpoint()
        self.train_csv = 'openi_report_train.csv'
        self.val_csv = 'openi_report_val.csv'
        self.test_csv = 'openi_report_test.csv'
        self.encoder_resume = self.cfg["RESUME_ENCODER"]
        self.decoder_resume_F = self.cfg["RESUME_DECODER_F"]
        self.decoder_resume_L = self.cfg["RESUME_DECODER_L"]
        self.D_resume = self.cfg["RESUME_D"]

        self.text_csv = self.cfg["TEXT_CSV"]
        self.img_csv = self.cfg["IMG_CSV"]
        self.data_root = self.cfg["DATA_ROOT"]
        self.batch_size = self.cfg["BATCH_SIZE"]
        self.image_size = tuple(self.cfg["IMAGE_SIZE"])
        self.name = self.cfg["EXPER_NAME"]

        self.pix_loss_ratio = self.cfg["PIXEL_LOSS_RATIO"]
        self.adv_loss_ratio = self.cfg["ADV_LOSS_RATIO"]
        self.id_loss_ratio = self.cfg["ID_LOSS_RATIO"]

        self.checkpoint_epoch = self.cfg["CHECKPOINT_EPOCH"]
        self.G_step = self.cfg["G_initer"]
        self.D_step = self.cfg["D_initer"]
        self.beta1 = self.cfg["beta1"]
        self.word_dict = self.cfg["DICTIONARY"]
        self.DISP_FREQs = [10, 20, 30, 50]
        self.lambda_gp = [10.0,10.0,10.0,10.0]
        self.writer = SummaryWriter(os.path.join("runs",self.exp_name))

        self.DECODERS = {
            "baseDECODER":baseDecoder,
            "baseDECODERv2": baseDecoderv2,
            "baseDECODERv3": baseDecoderv3
        }
        self.P_DECODER = {
            "PDECODER":PDecoder,
            "PDECODERv2":PDecoderv2,
            "PDECODERv3": PDecoderv3
        }
        self.DISCRIMINATOR = {
            "baseDISCRIMINATOR": baseDiscriminator,
            "noconDISCRIMINATOR": noCon_Discriminator,
            "Patch": PatchDiscriminator,
            "SNDiscriminator":SNDiscriminator,
            "ResDISCRIMINATOR": ResDiscriminator,
            "PDISCRIMINATOR":PDiscriminator
        }
        self.dataset = {
            "OPENI":[OpeniDataset2_Hiachy,OpeniDataset_Siamese],
            "MIMIC-CXR":[MIMICDataset2_Hiachy, MIMICDataset_Siamese]
        }


        #################################################
        ################ Dataset Setup ##################
        #################################################
        self.t2i_dataset = self.dataset[self.cfg["DATASET"]][0](csv_txt=self.text_csv,
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

        self.sia_dataset = self.dataset[self.cfg["DATASET"]][1](csv_txt=self.text_csv,
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
        self.S_criterion = nn.BCELoss().to(self.device)

        #########################################
        ############ Network Init ###############
        #########################################

        self.base_size = 32
        self.P_ratio = int(np.log2(self.image_size[0]//self.base_size))
        self.base_ratio = int(np.log2(self.base_size))
        print("Number of Decoders",self.P_ratio+1)
        print("Number of Discriminator", self.P_ratio + 1)

        self.define_nets()
        self.encoder = nn.DataParallel(self.encoder,device_ids=self.gpus)
        self.decoder_L = nn.DataParallel(self.decoder_L,device_ids=self.gpus)
        self.decoder_F = nn.DataParallel(self.decoder_F, device_ids=self.gpus)
        self.embednet = nn.DataParallel(self.embednet, device_ids=self.gpus)
        self.load_model()

    def define_nets(self):

        self.encoder = HAttnEncoderv3(vocab_size=self.t2i_dataset.vocab_size,
                                                          embed_size=self.cfg["E_EMBED_SIZE"],
                                                          hidden_size=self.cfg["E_HIDEN_SIZE"],
                                                          max_len=[self.t2i_dataset.max_len_finding,
                                                                   self.t2i_dataset.max_len_impression],
                                                          unit=self.cfg["RNN_CELL"],
                                                          feature_base_dim=self.cfg["D_CHANNEL_SIZE"]
                                                          ).to(self.device)

        decoders_F = []
        first_decoder = self.DECODERS[self.cfg["DECODER"]](input_dim=self.cfg["D_CHANNEL_SIZE"],
                                                           feature_base_dim=self.cfg["D_CHANNEL_SIZE"],
                                                           uprate=self.base_ratio).to(self.device)
        # first_decoder.apply(init_weights)
        decoders_F.append(first_decoder)
        for i in range(1, self.P_ratio + 1):
            nf = 128
            pdecoder = self.P_DECODER[self.cfg["PDECODER"]](input_dim=self.cfg["D_CHANNEL_SIZE"],
                                                            feature_base_dim=nf).to(self.device)
            # pdecoder.apply(init_weights)
            decoders_F.append(pdecoder)

        self.decoder_F = MultiscaleDecoder(decoders_F)

        decoders_L = []
        first_decoder = self.DECODERS[self.cfg["DECODER"]](input_dim=self.cfg["D_CHANNEL_SIZE"],
                                                           feature_base_dim=self.cfg["D_CHANNEL_SIZE"],
                                                           uprate=self.base_ratio).to(self.device)
        # first_decoder.apply(init_weights)
        decoders_L.append(first_decoder)
        for i in range(1, self.P_ratio + 1):
            nf = 128
            pdecoder = self.P_DECODER[self.cfg["PDECODER"]](input_dim=self.cfg["D_CHANNEL_SIZE"],
                                                            feature_base_dim=nf).to(self.device)
            # pdecoder.apply(init_weights)
            decoders_L.append(pdecoder)

        self.decoder_L = MultiscaleDecoder(decoders_L).to(self.device)

        self.embednet = Classifinet(backbone='resnet18').to(self.device)


    def define_dataloader(self,layer_id):

        self.train_dataloader = DataLoader(self.trainset,
                                           batch_size=self.batch_size[layer_id],
                                           shuffle=True,
                                           num_workers=8,
                                           drop_last=False)
        self.val_dataloader = DataLoader(self.valset,
                                           batch_size=self.batch_size[layer_id],
                                           shuffle=True,
                                           num_workers=8,
                                            drop_last=False)
        self.S_dataloader = DataLoader(self.sia_dataset,
                                       batch_size=self.batch_size[layer_id],
                                       shuffle=False,
                                       num_workers=8,
                                       pin_memory=True,
                                       drop_last=False)

    def define_opt(self,layer_id):
        '''Define optimizer'''
        self.G_optimizer = torch.optim.Adam([{'params': self.encoder.parameters()}] +
                                            [{'params': self.decoder_F.parameters()}] +
                                            [{'params': self.decoder_L.parameters()}],
                                            lr=self.cfg["G_LR"][layer_id],betas=(self.beta1, 0.999))
        self.G_lr_scheduler = MultiStepLR(self.G_optimizer, milestones=self.cfg["LR_DECAY_EPOCH"][layer_id], gamma=0.2)


        self.D_optimizer = torch.optim.Adam([{'params': self.D_F.parameters()}] +
                                            [{'params': self.D_L.parameters()}]
                                            , lr=self.cfg["D_LR"][layer_id],betas=(self.beta1, 0.999))

        self.D_lr_scheduler = MultiStepLR(self.D_optimizer, milestones=self.cfg["LR_DECAY_EPOCH"][layer_id], gamma=0.2)
        self.S_optimizer = torch.optim.Adam(self.embednet.parameters(), lr=self.cfg["S_LR"], betas=(self.beta1, 0.999))
        self.S_lr_scheduler = StepLR(self.S_optimizer, step_size=self.cfg["LR_SIAMESE_DECAY_EPOCH"], gamma=0.2)


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

        if os.path.exists(self.encoder_resume):
            print("load checkpoint {}".format(self.encoder_resume))
            self.encoder.load_state_dict(torch.load(self.encoder_resume))
        else:
            print("checkpoint do not exists {}".format(self.encoder_resume))

        if os.path.exists(self.decoder_resume_F):
            print("load checkpoint {}".format(self.decoder_resume_F))
            self.decoder_F.load_state_dict(torch.load(self.decoder_resume_F))
        else:
            print("checkpoint do not exists {}".format(self.decoder_resume_F))

        if os.path.exists(self.decoder_resume_L):
            print("load checkpoint {}".format(self.decoder_resume_L))
            self.decoder_L.load_state_dict(torch.load(self.decoder_resume_L))
        else:
            print("checkpoint do not exists {}".format(self.decoder_resume_L))



    def define_D(self,layer_id):
        '''Initialize a series of Discriminator'''

        dr = self.base_ratio - 2 + layer_id
        self.D_F = self.DISCRIMINATOR[self.cfg["DISCRIMINATOR"]](base_feature= self.cfg["DIS_CHANNEL_SIZE"],
                                                                txt_input_dim= self.cfg["D_CHANNEL_SIZE"],
                                                                down_rate = dr).to(self.device)

        self.D_L = self.DISCRIMINATOR[self.cfg["DISCRIMINATOR"]](base_feature=self.cfg["DIS_CHANNEL_SIZE"],
                                                               txt_input_dim=self.cfg["D_CHANNEL_SIZE"],
                                                               down_rate=dr).to(self.device)
        # self.D.apply(init_weights)
        if self.num_gpus > 1:
            self.D_F = nn.DataParallel(self.D_F,device_ids=self.gpus)
            self.D_L = nn.DataParallel(self.D_L,device_ids=self.gpus)

    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def Loss_on_layer(self,image, finding, impression,layer_id,decoder):
        '''
        Pretrain genertaor with batch
        :image image batch
        :text text batch
        '''

        txt_emded, hidden = self.encoder(finding, impression)
        r_image = F.interpolate(image, size=(2 ** layer_id) * self.base_size)

        self.G_optimizer.zero_grad()
        pre_image = decoder(txt_emded,layer_id)
        loss = self.G_criterion(pre_image.float(), r_image.float())
        loss.backward()
        self.G_optimizer.step()
        return loss,pre_image,r_image


    def train_Siamese_layer(self,layer_id):
        DISP_FREQ = self.DISP_FREQs[layer_id]

        for epoch in range(self.S_max_epoch[layer_id]):
            self.embednet.train()
            print('VCN Epoch [{}/{}]'.format(epoch, self.S_max_epoch[layer_id]))
            for idx, batch in enumerate(tqdm(self.S_dataloader)):
                image_f = batch['image_F'].to(self.device)
                image_l = batch['image_L'].to(self.device)
                label = batch['label'].to(self.device)

                r_image_f = F.interpolate(image_f, size=(2 ** layer_id) * self.base_size)
                r_image_l = F.interpolate(image_l, size=(2 ** layer_id) * self.base_size)

                self.S_optimizer.zero_grad()
                pred = self.embednet(r_image_f, r_image_l)
                loss = self.S_criterion(pred, label)
                loss.backward()
                self.S_optimizer.step()

                # if ((idx + 1) % DISP_FREQ == 0) and idx != 0:

                self.writer.add_scalar('Train_Siamese {}_loss'.format(layer_id),
                                       loss.item(),
                                       epoch * len(self.S_dataloader) + idx)
                self.writer.add_images("Train_front_{}_Original".format(layer_id),
                                       deNorm(r_image_f),
                                       epoch * len(self.S_dataloader) + idx)
                self.writer.add_images("Train_lateral_{}_Original".format(layer_id),
                                       deNorm(r_image_l),
                                       epoch * len(self.S_dataloader) + idx)

            self.S_lr_scheduler.step(epoch)

            self.embednet.eval()
            total = 0
            correct = 0
            for idx, batch in enumerate(tqdm(self.S_dataloader)):
                image_f = batch['image_F'].to(self.device)
                image_l = batch['image_L'].to(self.device)
                label = batch['label'].to(self.device)
                r_image_f = F.interpolate(image_f, size=(2 ** layer_id) * self.base_size)
                r_image_l = F.interpolate(image_l, size=(2 ** layer_id) * self.base_size)

                pred = self.embednet(r_image_f, r_image_l)
                pred[pred>0.5]=1
                pred[pred<=0.5]=0

                total += pred.shape[0]
                correct += torch.sum(pred==label).item()

            acc = correct / total
            # acc = self.evaluate_Siamese(layer_id)

            print(print("Accuracy {}".format(acc)))
            self.writer.add_scalar('Acc_Siamese_Layer {}'.format(layer_id),
                                   acc,
                                   epoch)

    def Loss_on_layer_GAN(self,image,finding, impression,layer_id,decoder,D):
        '''
        Pretrain genertaor with batch
        :image image batch
        :text text batch
        '''

        image = F.interpolate(image, size=(2 ** layer_id) * self.base_size)
        txt_emded, hidden = self.encoder(finding, impression)
        pre_image = decoder(txt_emded, layer_id)

        # Train Discriminator
        for _ in range(self.D_step):
            self.D_optimizer.zero_grad()
            pre_fake = D(pre_image, txt_emded)
            pre_real = D(image, txt_emded)
            gradient_penalty, gradients = cal_gradient_penalty(D, image, pre_image, txt_emded, "cuda",lambda_gp=self.lambda_gp[layer_id])

            D_loss = pre_fake.mean() - pre_real.mean() + gradient_penalty
            D_loss.backward(retain_graph=True)

            self.D_optimizer.step()

        # Train Generator
        for _ in  range(self.G_step):
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


    def train_layer(self,layer_id):
        DISP_FREQ = self.DISP_FREQs[layer_id]
        for epoch in range(20):
            print('Generator Epoch [{}/20]'.format(epoch))
            self.encoder.train()
            self.decoder_F.train()
            self.decoder_L.train()
            for idx, batch in enumerate(tqdm(self.train_dataloader)):
                finding = batch['finding'].to(self.device)
                impression = batch['impression'].to(self.device)
                image_f = batch['image_F'].to(self.device)
                image_l = batch['image_L'].to(self.device)

                loss_f, pre_image_f, r_image_f = self.Loss_on_layer(image_f, finding, impression, layer_id, self.decoder_F)
                loss_l, pre_image_l, r_image_l = self.Loss_on_layer(image_l,  finding, impression, layer_id, self.decoder_L)


                # print('Loss: {:.4f}'.format(loss.item()))
                if ((idx + 1) % DISP_FREQ == 0) and idx != 0:

                    self.writer.add_scalar('Train_front {}_loss'.format(layer_id),
                                           loss_f.item(),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_scalar('Train_lateral {}_loss'.format(layer_id),
                                           loss_l.item(),
                                           epoch * len(self.train_dataloader) + idx)

                    # write to tensorboard
                    self.writer.add_images("Train_front_{}_Original".format(layer_id),
                                           deNorm(r_image_f),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("Train_front_{}_Predicted".format(layer_id),
                                           deNorm(pre_image_f),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("Train_lateral_{}_Original".format(layer_id),
                                           deNorm(r_image_l),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("Train_lateral_{}_Predicted".format(layer_id),
                                           deNorm(pre_image_l),
                                           epoch * len(self.train_dataloader) + idx)

            self.G_lr_scheduler.step(epoch)

    def train_GAN_layer(self,layer_id):
        DISP_FREQ = self.DISP_FREQs[layer_id]
        self.encoder.train()
        self.decoder_F.train()
        self.decoder_L.train()
        self.D_F.train()
        self.D_L.train()
        for epoch in range(self.max_epoch[layer_id]):
            print('GAN Epoch [{}/{}]'.format(epoch,self.max_epoch[layer_id]))

            for idx, batch in enumerate(tqdm(self.train_dataloader)):
                finding = batch['finding'].to(self.device)
                impression = batch['impression'].to(self.device)
                image_f = batch['image_F'].to(self.device)
                image_l = batch['image_L'].to(self.device)

                D_loss_f, G_loss_f, pre_image_f, image_f = self.Loss_on_layer_GAN(image_f, finding, impression, layer_id, self.decoder_F,self.D_F)
                D_loss_l, G_loss_l, pre_image_l, image_l = self.Loss_on_layer_GAN(image_l, finding, impression, layer_id, self.decoder_L,self.D_L)

                # train with view consistency loss
                self.G_optimizer.zero_grad()
                pred = self.embednet(pre_image_f,pre_image_l)
                id_loss = self.id_loss_ratio * self.S_criterion(pred,torch.zeros_like(pred).to(self.device))
                id_loss.backward()
                self.G_optimizer.step()

                if ((idx + 1) % DISP_FREQ == 0) and idx != 0:
                    # ...log the running loss
                    # self.writer.add_scalar("Train_{}_SSIM".format(layer_id), ssim.ssim(r_image, pre_image).item(),
                    #                        epoch * len(self.train_dataloader) + idx)
                    self.writer.add_scalar('GAN_G_train_Layer_front_{}_loss'.format(layer_id),
                                           G_loss_f.item(),
                                      epoch * len(self.train_dataloader) + idx)
                    self.writer.add_scalar('GAN_D_train_Layer_front_{}_loss'.format(layer_id),
                                           D_loss_f.item(),
                                           epoch * len(self.train_dataloader) + idx)

                    self.writer.add_scalar('GAN_G_train_Layer_lateral_{}_loss'.format(layer_id),
                                           G_loss_l.item(),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_scalar('GAN_D_train_Layer_lateral_{}_loss'.format(layer_id),
                                           D_loss_l.item(),
                                           epoch * len(self.train_dataloader) + idx)
                    # write to tensorboard
                    self.writer.add_images("GAN_Train_Original_front_{}".format(layer_id),
                                           deNorm(image_f),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("GAN_Train_Predicted_front_{}".format(layer_id),
                                           deNorm(pre_image_f),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("GAN_Train_Original_lateral_{}".format(layer_id),
                                           deNorm(image_l),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("GAN_Train_Predicted_lateral_{}".format(layer_id),
                                           deNorm(pre_image_l),
                                           epoch * len(self.train_dataloader) + idx)
            self.G_lr_scheduler.step(epoch)
            self.D_lr_scheduler.step(epoch)
            if (epoch+1) % 20 == 0 and epoch != 0:
                torch.save(self.encoder.state_dict(), os.path.join(self.encoder_checkpoint,
                                                                   "Encoder_{}_Layer_{}_Time_{}_checkpoint.pth".format(
                                                                       self.cfg["ENCODER"], layer_id,
                                                                       get_time())))
                torch.save(self.D_F.state_dict(), os.path.join(self.D_checkpoint,
                                                               "D_{}_F_Layer_{}_Time_{}_checkpoint.pth".format(
                                                                   self.cfg["DISCRIMINATOR"], layer_id, get_time())))
                torch.save(self.D_L.state_dict(), os.path.join(self.D_checkpoint,
                                                               "D_{}_L_Layer_{}_Time_{}_checkpoint.pth".format(
                                                                   self.cfg["DISCRIMINATOR"], layer_id, get_time())))

                torch.save(self.decoder_F.state_dict(), os.path.join(self.decoder_checkpoint,
                                                                     "Decoder_{}_F_Layer_{}_Time_{}_checkpoint.pth".format(
                                                                         self.cfg["DECODER"], layer_id, get_time())))

                torch.save(self.decoder_L.state_dict(), os.path.join(self.decoder_checkpoint,
                                                                     "Decoder_{}_L_Layer_{}_Time_{}_checkpoint.pth".format(
                                                                         self.cfg["DECODER"], layer_id, get_time())))



    def train(self):

        # self.load_model()

        for layer_id in range(3,self.P_ratio+1):

            self.define_D(layer_id)
            self.define_dataloader(layer_id)
            self.define_opt(layer_id)

            #########################################################
            ############### Train Siamese by layer ################
            #########################################################
            print("Star training on Siamese {}".format(layer_id))
            self.train_Siamese_layer(layer_id)

            #########################################################
            ############### Train Generator by layer ################
            #########################################################
            if layer_id==0:

                print("Star training on Decoder {}".format(layer_id))
                self.train_layer(layer_id)

            #########################################################
            ################## Train GAN by layer ###################
            #########################################################
            print("Star training GAN {}".format(layer_id))
            self.train_GAN_layer(layer_id)




    def pare_cfg(self,cfg_json):
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
