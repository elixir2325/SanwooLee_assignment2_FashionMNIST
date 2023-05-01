#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhanghuangzhao
"""


import torch
from torch import nn
from torchvision.models import  resnet18, resnet34, resnet50, convnext_small

class CNN(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(1, 24, 5, 1, 2),
                nn.ReLU(),
                nn.Conv2d(24, 48, 5, 2, 2),
                nn.ReLU(),
                nn.Conv2d(48, 64, 5, 3, 2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(5 * 5 * 64, 200),
                nn.ReLU(),
                nn.Linear(200, 10)
            )

    def forward(self, x):
        
        x = x.reshape((-1, 1, 28, 28))

        return self.layers(x)


class SimpleResNet(nn.Module):
    def __init__(self):

        super().__init__()
        self.encoder =  resnet34(weights="DEFAULT")
        # there is only one channel for MNIST image
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.classifier = nn.Sequential(nn.ReLU(), nn.Dropout(0.2), nn.Linear(1000, 300), nn.ReLU(), nn.Dropout(0.2), nn.Linear(300, 10))

    def forward(self, x):
        x = x.reshape((-1, 1, 28, 28))
        x = self.encoder(x)
        # logits : (batch_size, class_n)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    
    from fmnist_dataset import load_fashion_mnist
    from torch.utils.data import DataLoader

    train, dev, test = load_fashion_mnist("../data")
    train_dataloader = DataLoader(train, batch_size=1)
    
    m = CNN()
    
    for x, y in train_dataloader:
        
        l = m(x)
        break