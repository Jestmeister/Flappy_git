
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

        self.fc1 = nn.Linear(4, 20)
        #self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        y = torch.div(x, torch.sum(x))

        return y



class AgentPG:
    def StartAgent(self, learning_rate, gamma, start_difficulty):
        self.env = environment.Game()
        self.value = value.Value(learning_rate)
        self.policyNet = PolicyNet()
        self.optimizer = torch.optim.RMSprop(self.policyNet.parameters(), lr=learning_rate)

        self.gamma = gamma #use in: DiscountedReward()
        self.start_difficulty = start_difficulty

        self.state = torch.empty(0, 4, dtype=torch.float32)
        self.reward = torch.empty(0, 1, dtype=torch.float32)
        self.discountedReward = torch.empty(0, 1, dtype=torch.float32)
        self.action = torch.empty(0, 1, dtype=torch.float32)

    def StartEnv(self):
        startState = self.env.Start(True, False, self.start_difficulty)
        self.state = torch.cat((self.state, startState), 0)

    

    #plays one frame of the game and saves the state, reward and action for UpdatePolicy()
    def Update(self):
        currentAction = self.SelectAction(self.state[len(self.state) - 1])

        currentState, currentReward, gameover = self.env.Update(currentAction)

        if not(gameover):
            self.state = torch.cat((self.state, currentState), 0)
        self.reward = torch.cat((self.reward, currentReward), 0)

        return not(gameover) #running

    #takes the current state as a torch tensor and returns true or false aka jump or not
    def SelectAction(self, currentState):
        actionProbabilityDis = self.policyNet(currentState)

        actionProbability = actionProbabilityDis.view(1, 1)
        self.action = torch.cat((self.action, actionProbability), 0)

        selection = random.random()
        if selection < actionProbabilityDis.item():
            return True
        else:
            return False
    


    def UpdatePolicy(self):
        print('Avg reward for this batch: {}'.format(torch.mean(self.reward).item()))
        self.DiscountedReward()
        self.value.UpdateValueNet(self.discountedReward, self.state)
        #print(self.state)
        #print(self.reward)
        #print(self.discountedReward)
        #print(self.action)

        loss = self.Loss(self.action)
        
        loss.backward()

        self.optimizer.step()

        self.policyNet.zero_grad()

        #resets the (training) data
        self.state = torch.empty(0, 4, dtype=torch.float32)
        self.reward = torch.empty(0, 1, dtype=torch.float32)
        self.discountedReward = torch.empty(0, 1, dtype=torch.float32)
        self.action = torch.empty(0, 1, dtype=torch.float32)

        #+ print loss function before and after update so one ses that its "improving"
        for param in self.policyNet.parameters():
            print(param.data)

    #calculates the discounted reward from reward and appends it to discountedReward
    def DiscountedReward(self):
        for i in range(len(self.reward)):
            currentDiscountedReward = 0
            for j in range(len(self.reward) - i):
                currentDiscountedReward += (self.gamma**j) * self.reward[j + i].item()

            currentDiscountedRewardTensor = torch.tensor([[currentDiscountedReward]], dtype=torch.float32)
            self.discountedReward = torch.cat((self.discountedReward, currentDiscountedRewardTensor), 0)

    def Loss(self, actions):
        #log(policy)A

        #print(self.discountedReward)
        #print(self.value.GetValue(self.state))
        A = self.discountedReward - self.value.GetValue(self.state)
        A = A.detach()
        #print(A)
        #print("")
        loss = torch.log(actions) * A
        #print(loss)
        loss = torch.mean(loss)
        #print(loss)
        #print(loss.grad)

        print('Total loss for this batch: {}'.format(loss.item()))

        return loss
