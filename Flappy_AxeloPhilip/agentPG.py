
from enum import Flag
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count
import matplotlib.pyplot as plt
import numpy as np

import random

#import gym
import environment



class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, 1)  # Prob of Left

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x



class AgentPG:
    def StartAgent(self, learning_rate, gamma, start_difficulty):
        pass #shuld set up all aribles needed for the runs ex. learning_rate and gamma
    
    def StartEnv(self):
        pass #shuld start/restart env
    
    def Update(self):
        pass #shuld use NN to play game and save actions, game states rewrds for each update
    
    def UpdatePolicy(self):
        pass #shuld run normal supervised lerning using the rewards, states and actions