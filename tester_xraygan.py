import torch
import json
from models.Encoder import baseEncoder, baseEncoderv2, harchyEncoder,HAttnEncoder, HAttnEncoderv2, HAttnEncoderv3
from models.Decoder import baseDecoder, PDecoderv2, baseDecoderv2, PDecoder,PDecoderv3, baseDecoderv3, MultiscaleDecoder
from models.Discriminator import SNDiscriminator, baseDiscriminator, noCon_Discriminator, PatchDiscriminator, \
    ResDiscriminator, PDiscriminator
from torch import nn

from utils.MIMICDataSet import MIMICDataset2_Hiachy
from utils.OpeniDataSet import OpeniDataset2_Hiachy
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from utils import get_time, matplotlib_imshow, deNorm, Rescale, ToTensor, Equalize
from tqdm import tqdm

from torchvision.utils import save_image
import os
import numpy as np



class Tester:
    def __init__(self):
        self.cfg_json = "config/openi_progressive_hiach_test.json"
        self.cfg = self.pare_cfg(self.cfg_json)
        self.exp_name = self.cfg["EXPER_NAME"]
        self.test_csv = 'openi_report_test.csv'
        self.encoder_resume = self.cfg["RESUME_ENCODER"]
        self.decoder_resume_F = self.cfg["RESUME_DECODER_F"]
        self.decoder_resume_L = self.cfg["RESUME_DECODER_L"]
        self.word_dict = self.cfg["DICTIONARY"]
        self.text_csv = self.cfg["TEXT_CSV"]
        self.img_csv = self.cfg["IMG_CSV"]
        self.data_root = self.cfg["DATA_ROOT"]
        self.image_size = tuple(self.cfg["IMAGE_SIZE"])
        self.name = self.cfg["EXPER_NAME"]
        self.save_img_dir = './save_image/OPENI_HIAv3'
        if os.path.exists(self.save_img_dir)==False:
            os.mkdir(self.save_img_dir)

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
        self.P_DECODER = {
            "PDECODER": PDecoder,
            "PDECODERv2": PDecoderv2,
            "PDECODERv3": PDecoderv3
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
            "OPENI": OpeniDataset2_Hiachy,
            "MIMIC-CXR": MIMICDataset2_Hiachy
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
                                         num_workers=8,
                                         drop_last=True)

        self.base_size = 32
        self.P_ratio = int(np.log2(self.image_size[0] // self.base_size))
        self.base_ratio = int(np.log2(self.base_size))
        print("Number of Decoders", self.P_ratio + 1)
        print("Number of Discriminator", self.P_ratio + 1)

        #########################################
        ############ Network Init ###############
        #########################################
        self.define_nets()
        if self.num_gpus > 1:
            self.encoder = nn.DataParallel(self.encoder, device_ids=self.gpus)
            self.decoder_L = nn.DataParallel(self.decoder_L, device_ids=self.gpus)
            self.decoder_F = nn.DataParallel(self.decoder_F, device_ids=self.gpus)

    def define_nets(self):
        self.encoder = HAttnEncoderv3(vocab_size=self.t2i_dataset.vocab_size,
                                                          embed_size=self.cfg["E_EMBED_SIZE"],
                                                          hidden_size=self.cfg["E_HIDEN_SIZE"],
                                                          max_len=[self.t2i_dataset.max_len_finding,
                                                                   self.t2i_dataset.max_len_impression],
                                                          unit=self.cfg["RNN_CELL"],
                                                          feature_base_dim=self.cfg["D_CHANNEL_SIZE"]
                                                          ).to(self.device)
        # self.encoder.apply(init_weights)
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

        self.decoder_L = MultiscaleDecoder(decoders_L)



    def load_model(self):
        self.encoder.load_state_dict(torch.load(self.encoder_resume))
        self.decoder_F.load_state_dict(torch.load(self.decoder_resume_F))
        self.decoder_L.load_state_dict(torch.load(self.decoder_resume_L))


    def test(self):
        self.load_model()
        layer_id = self.P_ratio
        self.encoder.eval()
        self.decoder_F.eval()
        self.decoder_L.eval()
        print("Start generating")
        for idx, batch in enumerate(tqdm(self.test_dataloader)):
            finding = batch['finding'].to(self.device)
            impression = batch['impression'].to(self.device)
            txt_emded, hidden = self.encoder(finding, impression)
            pre_image_f = self.decoder_F(txt_emded, layer_id)
            pre_image_l = self.decoder_L(txt_emded, layer_id)
            pre_image_f = deNorm(pre_image_f).data.cpu()
            pre_image_l = deNorm(pre_image_l).data.cpu()
            subject_id = batch['subject_id'].data.cpu().numpy()
            for i in range(pre_image_f.shape[0]):

                save_image(pre_image_f[i],'{}/{}_f.png'.format(self.save_img_dir,subject_id[i]))
                save_image(pre_image_l[i],'{}/{}_l.png'.format(self.save_img_dir,subject_id[i]))

    def save_origin(self):

        for idx, batch in enumerate(tqdm(self.test_dataloader)):
            image_f = batch['image_F'].to(self.device)
            image_l = batch['image_L'].to(self.device)
            image_f = deNorm(image_f).data.cpu()
            image_l = deNorm(image_l).data.cpu()
            subject_id = batch['subject_id'].data.cpu().numpy()
            for i in range(image_f.shape[0]):
                save_image(image_f[i], '{}/{}_f.png'.format(self.save_img_dir, subject_id[i]))
                save_image(image_l[i], '{}/{}_l.png'.format(self.save_img_dir, subject_id[i]))

    def pare_cfg(self, cfg_json):
        with open(cfg_json) as f:
            cfg = f.read()
            print(cfg)
            print("Config Loaded")
        return json.loads(cfg)


def main():
    trainer = Tester()
    trainer.test()


if __name__ == "__main__":
    main()
