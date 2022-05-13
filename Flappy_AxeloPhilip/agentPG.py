
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



#this shuld take a state and retunr a probility dis that termines the action
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, 2)  # Prob of Left

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.normalize(x)

        return x

#this shuld take in a state vector and estimate the disconnted reward based on the state
class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()

        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return x




class AgentPG:
    def StartAgent(self, learning_rate, gamma, start_difficulty):
        self.policyNet = PolicyNet()
        self.valueNet = ValueNet()
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)

        self.gamma = gamma #use in: DiscountedReward()

        self.state = []
        self.action = []
        self.reward = []
        self.discountedReward = []

        pass #shuld set up all aribles needed for the runs ex. learning_rate and gamma

    def StartEnv(self):
        pass #shuld start/restart env
    


    def Update(self):
        pass #shuld use NN to play game and save actions, game states rewrds for each update

    #takes the current state as a torch tensor and returns true or false aka jump or not
    def SelectAction(self, currentState):
        actionProbabilityDis = self.policyNet(currentState)

        selection = random.random()
        if selection > actionProbabilityDis[0].item():
            return True
        else:
            return False
            
    def DiscountedReward(self):
        pass #this shuld calculate the discounted reward from reward and append it to discountedReward
    


    def UpdatePolicy(self):
        pass #shuld run normal supervised lerning using the rewards, states and actions

    def UpdateValueNet(self):
        pass #this shuld update the ValueNet

    def Loss(self):
        return 0 #shuld be the avrege of: log(policy)A over the batch, A = discountedReward - ValueNet(state)