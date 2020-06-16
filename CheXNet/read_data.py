# encoding: utf-8

"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from utils.OpeniDataSet import read_png
class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, transform=None, only_f= False):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        # labels = []
        for image_name in os.listdir(data_dir):

            image_name = os.path.join(data_dir, image_name)
            if only_f and image_name.endswith('l.png'):
                pass
            else:
                # print(image_name)
                image_names.append(image_name)
                # labels.append(label)
        self.data_dir = data_dir
        self.image_names = image_names
        # self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        # label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image#, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

class ChestXrayDataSet_paired(Dataset):
    def __init__(self, data_dir1,data_dir2, transform=None, only_f= False):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names1 = []
        image_names2 = []
        # labels = []
        for image_name in os.listdir(data_dir1):

            image_name1 = os.path.join(data_dir1, image_name)
            image_name2 = os.path.join(data_dir2, image_name)
            if only_f and image_name.endswith('l.png'):
                pass
            else:
                # print(image_name)
                if os.path.exists(image_name1) and os.path.exists(image_name2):
                    image_names1.append(image_name1)
                    image_names2.append(image_name2)
                # labels.append(label)
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.image_names1 = image_names1
        self.image_names2 = image_names2
        # self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image
        """
        image_name1 = self.image_names1[index]
        image_name2 = self.image_names2[index]
        image1 = Image.open(image_name1).convert('RGB')
        image2 = Image.open(image_name2).convert('RGB')
        # label = self.labels[index]
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1,image2#, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names1)

class ChestXrayDataSet_Sia(Dataset):
    def __init__(self, data_dir, transform=None, only_f= False):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.image_names_f = []
        self.image_names_l = []
        # labels = []
        for image_name in os.listdir(data_dir):
            if image_name.endswith('l.png'):
                image_name_l = os.path.join(data_dir, image_name)
                image_name_f = os.path.join(data_dir, image_name.replace('l.png','f.png'))

                self.image_names_f.append(image_name_f)
                self.image_names_l.append(image_name_l)
                # labels.append(label)

        # self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image
        """
        image_namef = self.image_names_f[index]
        image_namel = self.image_names_l[index]
        imagef = np.array(read_png(image_namef))
        imagel = np.array(read_png(image_namel))
        # label = self.labels[index]
        if self.transform:
            imagef = self.transform(imagef)
            imagel = self.transform(imagel)
        return torch.tensor(imagef, dtype=torch.float),torch.tensor(imagel, dtype=torch.float)#, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names_f)

