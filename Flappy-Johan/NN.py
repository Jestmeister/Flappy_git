import random  # For generating random numbers
import sys  # We will use sys.exit to exit the program
import pygame
from pygame.locals import *  # Basic pygame imports
import time

import numpy as np
import math
import torch
from torch import nn


#Create the network
class DeepQNN(nn.Module):
    def __init__(self):
        super(DeepQNN, self).__init__()
        self.myNetwork = nn.Sequential(
            nn.Linear(6, 6),
            nn.Linear(6, 6),
            nn.Linear(6, 2),
        )

    def forward(self, x):
        qValues = self.myNetwork(x)
        return torch.sigmoid(qValues)
 
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DeepQNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print(model)

X = torch.rand(6, device=device)
X
X[0].item()
a=[X[0].item(),X[1].item()]
a=[1,2,3,4,5,6]
b=torch.FloatTensor(a)
Qvalues = model(b)
Qtarget=[Qvalues[0].item(),Qvalues[1].item()]

#Qvalues = model(torch.FloatTensor(FBoutput))
print(X)

for i in range(0,100):

    Qvalues = model(X)
    print(Qvalues)
    Qtarget=torch.FloatTensor([0.9,0.1])
    print(Qtarget)
    #Qtarget=torch.tensor([1.0,0.0])
    
    # Zero your gradients for every batch!
    optimizer.zero_grad()
    
    
    # Compute the loss and its gradients
    loss_fn = torch.nn.L1Loss()
    loss = loss_fn(Qvalues, Qtarget)
    loss.backward()
    optimizer.step()




print(X)
a=[]
for i in range(0,100):

    Qvalues = model(X)
    print(Qvalues)
    a.append(Qvalues)
    Qtarget=model(X)
    Qtarget[0]=1

    #Qtarget=torch.tensor([1.0,0.0])
    
    # Zero your gradients for every batch!
    optimizer.zero_grad()
    
    
    # Compute the loss and its gradients
    loss_fn = torch.nn.L1Loss()
    loss = loss_fn(Qvalues, Qtarget)
    loss.backward()
    optimizer.step()


x = torch.randn(3)
y = torch.sigmoid(x)








a=0.1
for i in range(0,20):
    print(a)
    a=a*1.3
    






















