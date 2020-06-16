from __future__ import print_function, division
import torch
import json
import pandas as pd
from torch.utils.data import Dataset
from utils.proprcessing import *
from tqdm import tqdm
import os
import random
import time


class OpeniDataset2(Dataset):
    """Biplane Text-to-image dataset for Open-i"""

    def __init__(self,
                 csv_txt,
                 csv_img,
                 root,
                 word_dict,
                 transform=None):
        """
        Args:
            csv_txt (string): Path to the csv file with Input txt.
            cvs_img (string): Path to the csv file with Label Images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.text_csv = pd.read_csv(csv_txt)
        self.img_csv = pd.read_csv(csv_img)
        self.root = root
        self.transform = transform
        if os.path.exists(word_dict):
            with open(word_dict) as f:
                self.word_to_idx, self.vocab_size, self.max_len_impression, self.max_len_finding = json.load(f)
        else:
            self.word_to_idx, self.vocab_size, self.max_len_impression, self.max_len_finding = self.get_word_idx()
            with open(word_dict,'w') as f:
                json.dump([self.word_to_idx, self.vocab_size, self.max_len_impression, self.max_len_finding], f)

        self.findings = []
        self.impression = []
        self.image_L = []
        self.image_F = []
        self.txt_len = []
        self.subject_ids = []
        print("Processing data.....")
        for index, row in tqdm(self.text_csv.iterrows()):
            subject_id = row['subject_id']
            txt_name = row['path']
            self.subject_ids.append(subject_id)

            fi, im = read_XML2(txt_name)

            finding = [self.word_to_idx[w] + 1 for w in fi]
            impression = [self.word_to_idx[w] + 1 for w in im]

            txt_finding = np.array(finding)
            txt_impression = np.array(impression)

            text_len = len(txt_finding)
            txt_finding = np.pad(txt_finding, (self.max_len_finding - text_len, 0), 'constant', constant_values=0)
            self.findings.append(txt_finding)
            text_len = len(impression)
            self.txt_len.append(text_len)
            txt_impression = np.pad(txt_impression, (self.max_len_impression - text_len, 0), 'constant',
                                    constant_values=0)
            self.impression.append(txt_impression)
            # Find the matching image for this report
            subject_imgs = self.img_csv[self.img_csv.subject_id == subject_id]

            img_name_L = subject_imgs[subject_imgs.direction == 'L'].iloc[0]['path']
            # For png data, load data and normalize
            self.image_L.append(img_name_L)

            # Find the matching image for this report
            img_name_F = subject_imgs[subject_imgs.direction == 'F'].iloc[0]['path']
            self.image_F.append(img_name_F)
    def __len__(self):
        return len(self.text_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # For png data, load data and normalize
        img_name_F = self.image_F[idx]
        img_name_L = self.image_L[idx]
        chest_img_F = np.array(read_png(img_name_F))
        chest_img_L = np.array(read_png(img_name_L))

        if self.transform:
            chest_img_F = self.transform(chest_img_F)
            chest_img_L = self.transform(chest_img_L)

        sample = {
            'subject_id': torch.tensor(self.subject_ids[idx],dtype=torch.long),
            'finding': torch.tensor(self.findings[idx],dtype=torch.long),
            'impression': torch.tensor(self.impression[idx],dtype=torch.long),
            'image_F': torch.tensor(chest_img_F,dtype=torch.float),
            'image_L': torch.tensor(chest_img_L,dtype=torch.float),
            'len': torch.tensor(self.txt_len[idx],dtype=torch.long)
        }
        return sample

    def get_word_idx(self):
        print("Counting Vocabulary....")
        wordbag = []
        sen_len_finding = []
        sen_len_impression = []
        for idx in tqdm(range(self.__len__())):
            txt_name = self.text_csv.iloc[idx]['path']
            fi, im = read_XML2(txt_name)
            sen_len_finding.append(len(fi))
            sen_len_impression.append(len(im))
            wordbag = wordbag + fi + im
        vocab = set(wordbag)
        word_to_idx = {}
        count = 0
        for i, word in enumerate(vocab):
            if word in word_to_idx.keys():
                pass
            else:
                word_to_idx[word] = count
                count += 1
        vocab_len = count + 1
        max_len_im,max_len_fi = max(sen_len_impression), max(sen_len_finding)
        print("Totally {} medical report".format(self.__len__()))
        print("Totally {} vocabulary".format(vocab_len))
        print("Max Finding length {}".format(max_len_fi))
        print("Max Impression length {}".format(max_len_im))
        return word_to_idx, vocab_len, max_len_im,max_len_fi

class OpeniDataset2_Hiachy(Dataset):
    """Biplane hierarchical Text-to-image dataset for Open-i"""

    def __init__(self,
                 csv_txt,
                 csv_img,
                 root,
                 word_dict,
                 transform=None):
        """
        Args:
            csv_txt (string): Path to the csv file with Input txt.
            cvs_img (string): Path to the csv file with Label Images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.text_csv = pd.read_csv(csv_txt)
        self.img_csv = pd.read_csv(csv_img)
        self.root = root
        self.transform = transform
        word_dict = 'hia_'+word_dict
        if os.path.exists(word_dict):
            with open(word_dict) as f:
                self.word_to_idx, self.vocab_size, self.max_len_impression, self.max_len_finding, self.max_word_len_impression, self.max_word_len_finding = json.load(f)
        else:
            self.word_to_idx, self.vocab_size, self.max_len_impression, self.max_len_finding,self.max_word_len_impression, self.max_word_len_finding = self.get_word_idx()
            with open(word_dict,'w') as f:
                json.dump([self.word_to_idx, self.vocab_size, self.max_len_impression, self.max_len_finding,self.max_word_len_impression, self.max_word_len_finding], f)

        self.findings = []
        self.impression = []
        self.image_L = []
        self.image_F = []
        self.subject_ids = []
        print("Processing data.....")
        for index, row in tqdm(self.text_csv.iterrows()):
            subject_id = row['subject_id']
            txt_name = row['path']
            self.subject_ids.append(subject_id)

            fi, im = read_XML_sentence(txt_name)

            txt_finding = []
            txt_impression = []

            for sen in fi:
                # print(sen)
                txt_finding_sen = [self.word_to_idx[w] for w in sen]
                txt_finding_sen = np.pad(txt_finding_sen, (self.max_word_len_finding - len(txt_finding_sen), 0),
                                         'constant', constant_values=0)
                txt_finding.append(txt_finding_sen)

            for sen in im:
                txt_impression_sen = [self.word_to_idx[w] for w in sen]
                txt_impression_sen = np.pad(txt_impression_sen,
                                            (self.max_word_len_impression - len(txt_impression_sen), 0), 'constant',
                                            constant_values=0)
                txt_impression.append(txt_impression_sen)

            txt_impression = np.pad(np.array(txt_impression),
                                    ((self.max_len_impression - len(txt_impression), 0), (0, 0)), 'constant',
                                    constant_values=0)
            txt_finding = np.pad(np.array(txt_finding), ((self.max_len_finding - len(txt_finding), 0), (0, 0)),
                                 'constant', constant_values=0)

            self.impression.append(txt_impression)
            self.findings.append(txt_finding)
            # Find the matching image for this report
            subject_imgs = self.img_csv[self.img_csv.subject_id == subject_id]

            img_name_L = subject_imgs[subject_imgs.direction == 'L'].iloc[0]['path']
            # For png data, load data and normalize
            self.image_L.append(img_name_L)

            # Find the matching image for this report
            img_name_F = subject_imgs[subject_imgs.direction == 'F'].iloc[0]['path']
            self.image_F.append(img_name_F)
    def __len__(self):
        return len(self.text_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # For png data, load data and normalize
        img_name_F = self.image_F[idx]
        img_name_L = self.image_L[idx]
        chest_img_F = np.array(read_png(img_name_F))
        chest_img_L = np.array(read_png(img_name_L))

        if self.transform:
            chest_img_F = self.transform(chest_img_F)
            chest_img_L = self.transform(chest_img_L)

        sample = {
            'subject_id': torch.tensor(self.subject_ids[idx],dtype=torch.long),
            'finding': torch.tensor(self.findings[idx],dtype=torch.long),
            'impression': torch.tensor(self.impression[idx],dtype=torch.long),
            'image_F': torch.tensor(chest_img_F,dtype=torch.float),
            'image_L': torch.tensor(chest_img_L,dtype=torch.float)
        }
        return sample

    def get_word_idx(self):
        print("Counting Vocabulary....")
        wordbag = []
        sen_len_finding = []
        sen_len_impression = []
        word_len_finding = []
        word_len_impression = []
        for idx in tqdm(range(self.__len__())):
            txt_name = self.text_csv.iloc[idx]['path']
            fi, im = read_XML_sentence(txt_name)
            sen_len_finding.append(len(fi))
            sen_len_impression.append(len(im))
            for sen in fi:
                word_len_finding.append(len(sen))
                wordbag += sen
            for sen in im:
                word_len_impression.append(len(sen))
                wordbag += sen

        vocab = set(wordbag)
        word_to_idx = {}
        count = 1
        for i, word in enumerate(vocab):
            if word in word_to_idx.keys():
                pass
            else:
                word_to_idx[word] = count
                count += 1
        vocab_len = count + 1
        max_len_im,max_len_fi = max(sen_len_impression), max(sen_len_finding)
        max_word_len_im,max_word_len_fi = max(word_len_impression), max(word_len_finding)
        print("Totally {} medical report".format(self.__len__()))
        print("Totally {} vocabulary".format(vocab_len))
        print("Max Finding: sent length {} \t word lenth {}".format(max_len_fi,max_word_len_fi))
        print("Max Impression: length {} \t word lenth {}".format(max_len_im, max_word_len_im))
        return word_to_idx, vocab_len, max_len_im,max_len_fi,max_word_len_im,max_word_len_fi


class OpeniDataset_Siamese(Dataset):
    """View consistency dataset for Open-i"""

    def __init__(self,
                 csv_txt,
                 csv_img,
                 root,
                 transform=None):
        """
        Args:
            csv_txt (string): Path to the csv file with Input txt.
            cvs_img (string): Path to the csv file with Label Images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.text_csv = pd.read_csv(csv_txt)
        self.img_csv = pd.read_csv(csv_img)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.text_csv)

    def get_one_data(self,idx):

        subject_id = self.text_csv.iloc[idx]['subject_id']

        # Find the matching image for this report
        subject_imgs = self.img_csv[self.img_csv.subject_id == subject_id]

        img_name_L = subject_imgs[subject_imgs.direction == 'L'].iloc[0]['path']
        # For png data, load data and normalize
        chest_img_L = np.array(read_png(img_name_L))

        # Find the matching image for this report
        img_name_F = subject_imgs[subject_imgs.direction == 'F'].iloc[0]['path']
        # For png data, load data and normalize
        chest_img_F = np.array(read_png(img_name_F))

        if self.transform:
            chest_img_F = self.transform(chest_img_F)
            chest_img_L = self.transform(chest_img_L)

        sample = {
            'image_F': torch.tensor(chest_img_F, dtype=torch.float),
            'image_L': torch.tensor(chest_img_L, dtype=torch.float),
        }
        return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.get_one_data(idx)
        p = random.uniform(0, 1)
        if p > 0.5:
            # Randomly choose a sample idx!=n_idx
            n_idx = idx
            while idx == n_idx:
                n_idx = random.randint(0, self.__len__() - 1)
            negative = self.get_one_data(n_idx)
            sample['image_L'] = negative['image_L']
            sample['label'] = torch.ones(1).float()
        else:
            sample['label'] = torch.zeros(1).float()
        return sample




