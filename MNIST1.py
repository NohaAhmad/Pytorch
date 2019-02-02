#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 20:21:02 2019

@author: noha
"""

import torch
import matplotlib.pyplot as plt
#torchvision contains popular datasets, model architectures and common image tranformations#
from torchvision import datasets, transforms
from torch import nn
from collections import OrderedDict
from torch import optim

input_size = 784
hidden_size = [128,64]
out_size = 10
epochs = 30
epochsList = [i for i in range(epochs)]
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = datasets.MNIST("~/.pytorch/MNIST_data/", download = True, train = True, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainset,batch_size = 32, shuffle = True)
#print(len(trainLoader))
model = nn.Sequential(OrderedDict([(
                'fc1', nn.Linear(input_size,hidden_size[0])),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(hidden_size[0],hidden_size[1])),
                ('relu2', nn.ReLU()),
                ('out',nn.Linear(hidden_size[1],out_size)),
                ('logSoftmax', nn.LogSoftmax(dim=1))]))
optimizer = optim.SGD(model.parameters(),lr=0.08)
criterion = nn.NLLLoss()
loss_list = []
for e in range(epochs):
        running_loss = 0
        for images,labels in trainLoader:
            images_process = images.resize(images.shape[0],784)
            logProbs = model(images_process)
            optimizer.zero_grad()
            loss = criterion(logProbs,labels)
            #print("Initial Weights", model.fc1.weight[0])
            loss.backward()
            #print("Gradient",model.fc1.weight.grad[0])
            optimizer.step()
            running_loss += loss.item()
        print("Training Loss = {}".format(running_loss/len(trainLoader)))
        loss_list.append(running_loss/len(trainLoader))
        #print("Updated weights",model.fc1.weight[0])
plt.plot(epochsList,loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training phase")