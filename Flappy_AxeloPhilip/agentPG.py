
import imp
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

import value



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
        
        sum = torch.sum(x)
        y = (1 / sum)*x
        
        return y



class AgentPG:
    def StartAgent(self, learning_rate, gamma, start_difficulty):
        self.env = environment.Game()
        self.value = value.Value(learning_rate)
        self.policyNet = PolicyNet()
        self.optimizer = torch.optim.RMSprop(self.policyNet.parameters(), lr=learning_rate)

        self.gamma = gamma #use in: DiscountedReward()
        self.start_difficulty = start_difficulty

        self.state = []

        self.reward = []
        self.discountedReward = []

        self.action = []

    def StartEnv(self):
        currentState = self.env.Start(True, False, self.start_difficulty)
        self.state.append(currentState)

    

    #plays one frame of the game and saves the state, reward and action for UpdatePolicy()
    def Update(self):
        currentAction = self.SelectAction(self.state[len(self.state) - 1])

        currentState, currentReward, gameover = self.env.Update(currentAction)
        self.state.append(currentState)
        self.reward.append(currentReward)

        return not(gameover) #running

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
        self.DiscountedReward()
        self.value.UpdateValueNet(self.discountedReward, self.state)

        loss = self.Loss()

        loss.backward()

        self.optimizer.step()

        self.zero_grad()

        #resets the (training) data
        self.state = []
        self.reward = []
        self.discountedReward = []
        self.action = []

        #+ print loss function before and after update so one ses that its "improving"

    #calculates the discounted reward from reward and appends it to discountedReward
    def DiscountedReward(self):
        for i in range(len(self.reward)):
            for j in range(len(self.reward) - i):
                currentDiscountedReward = (self.gamma**j) * self.reward[j + i]
                self.discountedReward.append(torch.tensor(currentDiscountedReward))

    def Loss(self):
        #all these shuld be the same:
        print(len(self.state))
        print(len(self.reward))
        print(len(self.discountedReward))
        print(len(self.action))

        #log(policy)A
        for i in range(len(self.state)):
            #A_t = G_t - V(t)
            A = self.discountedReward[i] - self.value.GetValue(self.state[i])

            loss += torch.log(self.action[i]) * A
        
        loss /= len(self.state)

        print('Total loss for this batch: {}'.format(loss.item()))

        return loss