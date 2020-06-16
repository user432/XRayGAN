from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from utils.proprcessing import *
from tqdm import tqdm
import json
import numpy as np
import random
import time

class MIMICDataset2(Dataset):
    """Biplane Text-to-image dataset for MIMIC"""

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
        csv_report = '../MIMIC/physionet.org/files/mimic-cxr/2.0.0/files/mimic_cxr_sectioned.csv'
        self.text_csv = pd.read_csv(csv_txt)
        self.img_csv = pd.read_csv(csv_img)
        self.report_csv = pd.read_csv(csv_report)
        print(self.report_csv.shape)
        self.root = root
        self.transform = transform
        if os.path.exists(word_dict):
            with open(word_dict) as f:
                self.word_to_idx, self.vocab_size, self.max_len_impression, self.max_len_finding = json.load(f)
        else:
            self.word_to_idx, self.vocab_size, self.max_len_impression, self.max_len_finding = self.get_word_idx()
            with open(word_dict,'w') as f:
                json.dump([self.word_to_idx, self.vocab_size, self.max_len_impression, self.max_len_finding], f)
    def __len__(self):
        return len(self.text_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        subject_id = self.text_csv.iloc[idx].subject_id
        subject_report = self.report_csv[self.report_csv.study == subject_id].iloc[0]
        raw_fi, raw_im = subject_report.findings.split(), subject_report.impression.split()
        fi = [normalizeString(s) for s in raw_fi]
        im = [normalizeString(s) for s in raw_im]

        finding = [self.word_to_idx[w]+1 for w in fi]
        impression = [self.word_to_idx[w]+1 for w in im]

        txt_finding = np.array(finding)
        text_len = len(txt_finding)
        txt_finding = np.pad(txt_finding, (self.max_len_finding - text_len, 0), 'constant', constant_values=0)

        txt_impression = np.array(impression)
        text_len = len(impression)
        txt_impression = np.pad(txt_impression, (self.max_len_impression - text_len, 0), 'constant', constant_values=0)

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
            'subject_id': torch.tensor(subject_id,dtype=torch.long),
            'finding': torch.tensor(txt_finding,dtype=torch.long),
            'impression': torch.tensor(txt_impression,dtype=torch.long),
            'image_F': torch.tensor(chest_img_F,dtype=torch.float),
            'image_L': torch.tensor(chest_img_L,dtype=torch.float),
            'len': torch.tensor(text_len,dtype=torch.long)
        }
        return sample

    def get_word_idx(self):
        print("Counting Vocabulary....")
        wordbag = []
        sen_len_finding = []
        sen_len_impression = []
        for idx in tqdm(range(self.__len__())):
            subject_id = self.text_csv.iloc[idx].subject_id
            subject_report = self.report_csv[self.report_csv.study==subject_id].iloc[0]
            raw_fi, raw_im =subject_report.findings.split(),subject_report.impression.split()
            fi = [normalizeString(s) for s in raw_fi]
            im = [normalizeString(s) for s in raw_im]
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

class MIMICDataset2_Hiachy(Dataset):
    """Biplane hierarchical Text-to-image dataset for MIMIC"""

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
        csv_report = '../MIMIC/physionet.org/files/mimic-cxr/2.0.0/files/mimic_cxr_sectioned.csv'
        word_dict = 'hia_' + word_dict
        self.text_csv = pd.read_csv(csv_txt)
        self.img_csv = pd.read_csv(csv_img)
        self.report_csv = pd.read_csv(csv_report)
        self.root = root
        self.transform = transform
        if os.path.exists(word_dict):
            with open(word_dict) as f:
                self.word_to_idx, self.vocab_size, self.max_len_impression, self.max_len_finding = json.load(f)
        else:
            self.word_to_idx, self.vocab_size, self.max_len_impression, self.max_len_finding = self.get_word_idx()
            with open(word_dict,'w') as f:
                json.dump([self.word_to_idx, self.vocab_size, self.max_len_impression, self.max_len_finding], f)
    def __len__(self):
        return len(self.text_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        subject_id = self.text_csv.iloc[idx].subject_id
        subject_report = self.report_csv[self.report_csv.study == subject_id].iloc[0]
        raw_fi, raw_im = subject_report.findings.split(), subject_report.impression.split()
        fi = [normalizeString(s) for s in raw_fi]
        im = [normalizeString(s) for s in raw_im]

        finding = [self.word_to_idx[w]+1 for w in fi]
        impression = [self.word_to_idx[w]+1 for w in im]

        txt_finding = np.array(finding)
        text_len = len(txt_finding)
        txt_finding = np.pad(txt_finding, (self.max_len_finding - text_len, 0), 'constant', constant_values=0)

        txt_impression = np.array(impression)
        text_len = len(impression)
        txt_impression = np.pad(txt_impression, (self.max_len_impression - text_len, 0), 'constant', constant_values=0)

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
            'subject_id': torch.tensor(subject_id,dtype=torch.long),
            'finding': torch.tensor(txt_finding,dtype=torch.long),
            'impression': torch.tensor(txt_impression,dtype=torch.long),
            'image_F': torch.tensor(chest_img_F,dtype=torch.float),
            'image_L': torch.tensor(chest_img_L,dtype=torch.float),
            'len': torch.tensor(text_len,dtype=torch.long)
        }
        return sample

    def get_word_idx(self):
        print("Counting Vocabulary....")
        wordbag = []
        sen_len_finding = []
        sen_len_impression = []
        for idx in tqdm(range(self.__len__())):
            subject_id = self.text_csv.iloc[idx].subject_id
            subject_report = self.report_csv[self.report_csv.study==subject_id].iloc[0]
            raw_fi, raw_im =subject_report.findings.split(),subject_report.impression.split()
            fi = [normalizeString(s) for s in raw_fi]
            im = [normalizeString(s) for s in raw_im]
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

class MIMICDataset_Siamese(Dataset):
    """Text-to-image dataset"""

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
                n_idx = random.randint(0,self.__len__()-1)
            negative = self.get_one_data(n_idx)
            sample['image_L'] = negative['image_L']
            sample['label'] = torch.ones(1).float()
        else:
            sample['label'] = torch.zeros(1).float()
        return sample

if __name__ == '__main__':
    dataset = MIMICDataset_Siamese(csv_txt='./config/MIMIC_p10_reports.csv',
                 csv_img='./config/MIMIC_p10_images.csv',
                 root='')
    for i in tqdm(range(len(dataset))):
        dataset.get_one_data(i)



