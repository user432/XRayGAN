import torch
import json
from models.Encoder import baseEncoder, baseEncoderv2, RNN_ENCODER
from models.AttnGAN import G_NET
from torch import nn
from utils.OpeniDataSet import OpeniDataset2
from utils.MIMICDataSet import MIMICDataset2
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from utils.proprcessing import get_time, matplotlib_imshow, deNorm, Rescale, ToTensor, Equalize
from tqdm import tqdm

from torchvision.utils import save_image
import os
import numpy as np



class Tester:
    def __init__(self):
        self.cfg_json = "config/MIMIC_Attn_test.json"
        self.cfg = self.pare_cfg(self.cfg_json)
        self.exp_name = self.cfg["EXPER_NAME"]

        self.encoder_resume = self.cfg["RESUME_ENCODER"]
        self.decoder_resume_F = self.cfg["RESUME_DECODER_F"]
        self.decoder_resume_L = self.cfg["RESUME_DECODER_L"]
        self.word_dict = self.cfg["DICTIONARY"]
        self.text_csv = self.cfg["TEXT_CSV"]
        self.img_csv = self.cfg["IMG_CSV"]
        self.data_root = self.cfg["DATA_ROOT"]
        self.image_size = tuple(self.cfg["IMAGE_SIZE"])
        self.name = self.cfg["EXPER_NAME"]

        self.test_csv = self.cfg["TEST_CSV"]
        self.save_img_dir1 = './save_image/MIMIC_Attn256'
        self.save_img_dir2 = './save_image/MIMIC_origin'


        self.ENCODERS = { "RNN_ENCODER" : RNN_ENCODER
        }

        self.dataset = {
            "OPENI": OpeniDataset2,
            "MIMIC": MIMICDataset2
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
        self.testset = self.dataset[self.cfg["DATASET"]](csv_txt=self.test_csv,
                                                             csv_img=self.img_csv,
                                                             root=self.data_root,
                                                             word_dict=self.word_dict,
                                                             transform=transforms.Compose([
                                                                 Rescale(self.image_size),
                                                                 Equalize(),
                                                                 ToTensor()
                                                             ]))

        self.testset = self.t2i_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        s_gpus = self.cfg["GPU_ID"].split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)


        self.test_dataloader = DataLoader(self.testset,
                                         batch_size=12,
                                         shuffle=False,
                                         num_workers=0,
                                         drop_last=True)

        self.base_size = self.image_size[0]
        self.base_ratio = int(np.log2(self.base_size))

        #########################################
        ############ Network Init ###############
        #########################################
        self.define_nets()

        self.decoder_L= self.define_nets()
        self.decoder_F= self.define_nets()
        self.encoder = self.ENCODERS[self.cfg["ENCODER"]](vocab_size=self.t2i_dataset.vocab_size,
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

        return netG

    def load_model(self):
        print("Model Loading.............")
        self.encoder.load_state_dict(torch.load(self.encoder_resume))
        self.decoder_F.load_state_dict(torch.load(self.decoder_resume_F))
        self.decoder_L.load_state_dict(torch.load(self.decoder_resume_L))


    def test(self):
        self.load_model()
        self.encoder.eval()
        self.decoder_F.eval()
        self.decoder_L.eval()
        print("Start generating")
        for idx, batch in enumerate(tqdm(self.test_dataloader)):
            finding = batch['finding'].to(self.device)
            impression = batch['impression'].to(self.device)
            words_emb, sent_emb = self.encoder(finding, impression)
            words_emb, sent_emb = words_emb.detach(), sent_emb.detach()
            mask1 = (finding == 0)
            mask2 = (impression == 0)
            mask = torch.cat((mask1, mask2), 1)
            num_words = words_emb.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]
            pre_image_f,_, mu, logvar = self.decoder_F(sent_emb, words_emb, mask)
            pre_image_l,_, mu, logvar = self.decoder_L(sent_emb, words_emb, mask)

            pre_image_f = deNorm(pre_image_f[-1]).data.cpu()
            pre_image_l = deNorm(pre_image_l[-1]).data.cpu()
            subject_id = batch['subject_id'].data.cpu().numpy()
            for i in range(pre_image_f.shape[0]):

                save_image(pre_image_f[i],'{}/{}_f.png'.format(self.save_img_dir1,subject_id[i]))
                save_image(pre_image_l[i],'{}/{}_l.png'.format(self.save_img_dir1,subject_id[i]))

    def save_origin(self):
        for idx, batch in enumerate(tqdm(self.test_dataloader)):
            image_f = batch['image_F'].to(self.device)
            image_l = batch['image_L'].to(self.device)
            image_f = deNorm(image_f).data.cpu()
            image_l = deNorm(image_l).data.cpu()
            subject_id = batch['subject_id'].data.cpu().numpy()
            for i in range(image_f.shape[0]):
                save_image(image_f[i], '{}/{}_f.png'.format(self.save_img_dir2, subject_id[i]))
                save_image(image_l[i], '{}/{}_l.png'.format(self.save_img_dir2, subject_id[i]))



    def pare_cfg(self, cfg_json):
        with open(cfg_json) as f:
            cfg = f.read()
            print(cfg)
            print("Config Loaded")
        return json.loads(cfg)


def main():
    trainer = Tester()
    trainer.test()
    trainer.save_origin()


if __name__ == "__main__":
    main()
