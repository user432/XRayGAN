import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import numpy as np
import random
from torchvision.datasets import MNIST
from efficientnet_pytorch import EfficientNet
from torch.nn.modules.distance import PairwiseDistance
class EmbeddingNet(nn.Module):
    def __init__(self,
                 backbone = "our",
                 embedding_size=128):
        super(EmbeddingNet, self).__init__()

        if backbone=="resnet18":
            pretrained_net = models.resnet18(pretrained=True)
            self.inplanes = 64

            self.convnet = nn.Sequential()
            self.convnet.add_module('conv1',nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False))
            for idx, layer in enumerate(pretrained_net.children()):
                # Change the first conv and last linear layer
                if isinstance(layer, nn.Linear) == False and idx!=0:
                    self.convnet.add_module(str(idx),layer)

            self.use_fc = True
            self.fc = nn.Linear(512, embedding_size)

        elif backbone=="efficientnet-b0":
            self.use_fc = False
            self.convnet = EfficientNet.from_pretrained('efficientnet-b0',num_classes=128, in_channels=1)
        elif backbone=="vgg11":

            pretrained_net = models.vgg11().features
            self.inplanes = 64

            self.convnet = nn.Sequential()
            self.convnet.add_module('conv1', nn.Conv2d(1, self.inplanes, kernel_size=3, padding=1))
            for idx, layer in enumerate(pretrained_net.children()):
                # Change the first conv and last linear layer
                if idx != 0:
                    self.convnet.add_module(str(idx), layer)
            self.convnet.add_module('averagepool',nn.AdaptiveAvgPool2d(1))
            self.use_fc = True
            self.fc = nn.Linear(512, embedding_size)

        elif backbone=="alex":
            self.use_fc = True

            self.convnet = nn.Sequential(
            # conv1
            nn.Conv2d(1, 96, 5, 1,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(96, 256, 5, 1,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(256, 384, 3, 1,padding=1),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 384, 3, 1,padding=1),
            nn.ReLU(inplace=True),
            # conv5
            nn.AdaptiveAvgPool2d(1))
            self.fc = nn.Linear(384, embedding_size)


    def forward(self, x):
        return self.get_embedding(x)

    def get_embedding(self, x):
        output = self.convnet(x)
        if self.use_fc:
            output = output.view(output.size()[0], -1)
            output = self.fc(output)
        output = F.normalize(output, p=2, dim=1)
        # multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        output = output * alpha
        return output

class Classifinet(EmbeddingNet):
    def __init__(self,
                 backbone="efficientnet-b0",
                 embedding_size=128):
        super(Classifinet, self).__init__(backbone,embedding_size)
        self.classifier = nn.Sequential(nn.Linear(embedding_size, 1))
        # self.initialize()

    def initialize(self):
        nn.init.xavier_uniform(self.classifier.weight.data)
        self.classifier.bias.data.zero_()

    def forward(self, x1, x2):
        embed_1 = self.get_embedding(x1)
        embed_2 = self.get_embedding(x2)

        embed = torch.abs(embed_1-embed_2)
        output = torch.sigmoid(self.classifier(embed))
        return output


class DTripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin = 5.):
        super(DTripletLoss, self).__init__()
        self.margin = margin
        self.pair_loss = PairwiseDistance(2)


    def forward(self, anchor1, anchor2, negative1, negative2, size_average=True):
        distance_positive1 = self.pair_loss(anchor1, anchor2)
        distance_positive2 = self.pair_loss(negative1, negative2)
        distance_negative1 = self.pair_loss(anchor1, negative2)
        distance_negative2 = self.pair_loss(anchor2, negative1)
        losses = distance_positive1 + distance_positive2 + F.relu( - distance_negative1 - distance_negative2 + self.margin)
        return losses.mean()

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin = 1.):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pair_loss = PairwiseDistance(2)
        self.alpha = 2


    def forward(self, embed_f, embed_l, size_average=True):
        batch_size = embed_f.shape[0]
        index = np.arange(batch_size)
        new_index = index.copy()-1
        distance_positive1 = self.pair_loss(embed_f[index], embed_l[index])
        distance_negative1 = self.pair_loss(embed_f[index], embed_l[new_index])
        distance_negative2 = self.pair_loss(embed_f[new_index], embed_l[index])

        losses = F.relu(2*distance_positive1 - distance_negative1 - distance_negative2 + self.margin)
        return losses.mean(), distance_positive1.mean(), (distance_negative2+distance_negative1).mean()/2

class ITripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin = 1.):
        super(ITripletLoss, self).__init__()
        self.margin = margin
        self.pair_loss = PairwiseDistance(2)


    def forward(self, embed_f, embed_l, size_average=True):
        split_index = embed_f.shape[0]//2
        distance_positive1 = self.pair_loss(embed_f[:split_index], embed_l[:split_index])
        distance_positive2 = self.pair_loss(embed_f[split_index:], embed_l[split_index:])
        distance_negative1 = self.pair_loss(embed_f[:split_index], embed_f[split_index:])
        distance_negative2 = self.pair_loss(embed_l[:split_index], embed_l[split_index:])
        losses =  F.relu(distance_positive1 + distance_positive2 - distance_negative1 - distance_negative2 + self.margin)
        return losses.mean()

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin = 1.):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_loss = PairwiseDistance(2)


    def forward(self, embed_f, embed_l, size_average=True):
        losses = self.pair_loss(embed_f, embed_l)
        return losses.mean()

class ClassificationLoss(nn.Module):
    """
       Contrastive loss
       Takes embeddings of an anchor sample, a positive sample and a negative sample
       """
    def __init__(self,embed_size=128):
        super(ClassificationLoss, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(embed_size*2,1),
                                        nn.Sigmoid())

        self.cretio = nn.BCELoss()
        # self.initialize()



    def forward(self,embed_f, embed_l, target):
        pred = self.predict(embed_f,embed_l)
        print(pred)
        print(target)
        loss = self.cretio(pred,target)
        return loss

    def evaluate(self,embed_f, embed_l):
        batch_size = embed_f.shape[0]
        index = np.arange(batch_size)
        new_index = index.copy() - 1

        correct = 0
        correct += torch.sum(self.predict(embed_f[index],embed_l[index])>0.5).data.cpu().numpy()
        correct += torch.sum(self.predict(embed_f[index],embed_l[new_index])<0.5).data.cpu().numpy()
        return batch_size*2,correct

    def predict(self,embed_1, embed_2):
        input_vect = torch.cat([embed_1,embed_2],dim=1)
        return self.classifier(input_vect)







