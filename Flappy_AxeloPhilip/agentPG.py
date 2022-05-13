
from enum import Flag
from math import gamma
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
        self.start_difficulty = start_difficulty

        self.state = []

        self.reward = []
        self.discountedReward = []

        self.action = []

        pass #shuld set up all aribles needed for the runs ex. learning_rate and gamma

    def StartEnv(self):
        currentState = self.env.Start(True, False, self.start_difficulty)
        self.state.append(currentState)

    

    #plays one frame of the game and saves the state, reward and action for UpdatePolicy()
    def Update(self):
        currentAction = self.SelectAction(self.state[len(self.state) - 1])

        currentState, currentReward = self.env.Update(currentAction)
        self.state.append(currentState)
        self.reward.append(currentReward)

    #takes the current state as a torch tensor and returns true or false aka jump or not
    def SelectAction(self, currentState):
        actionProbabilityDis = self.policyNet(currentState)

        selection = random.random()
        if selection < actionProbabilityDis[0].item():
            self.action.append(actionProbabilityDis[0])

            return True
        else:
            self.action.append(actionProbabilityDis[1])

            return False
    


    def UpdatePolicy(self):
        pass #shuld run normal supervised lerning using the rewards, states and actions

        #resats the (training) data
        self.state = []
        self.reward = []
        self.discountedReward = []
        self.action = []

    def UpdateValueNet(self):
        self.DiscountedReward()

        pass #this shuld update the ValueNet

    #calculates the discounted reward from reward and appends it to discountedReward
    def DiscountedReward(self):
        for i in range(len(self.reward)):
            for j in range(len(self.reward) - i):
                currentDiscountedReward = (self.gamma**j) * self.reward[j + i]
                self.discountedReward.append(currentDiscountedReward)

    def Loss(self):
        return 0 #shuld be the avrege of: log(policy)A over the batch, A = discountedReward - ValueNet(state)