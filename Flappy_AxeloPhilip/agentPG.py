
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
    def Start(self, learning_rate, start_difficulty):
        self.env = environment.Game()
        self.policy_net = PolicyNet()
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
        self.difficulty = start_difficulty
        # Batch History
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []
        self.steps = 0

        self.episode_durations = []
    
    def StartEnv(self):
        self.state = self.env.Start(True, False, self.difficulty)
        self.state = torch.from_numpy(self.state).float()
        self.state = Variable(self.state)
        self.t = 0
    
    def Update(self):
        probs = self.policy_net(self.state)
        m = Bernoulli(probs)
        action = m.sample()

        action = action.data.numpy().astype(int)[0]
        next_state, reward, done = self.env.Update(action)
        
        # To mark boundarys between episodes
        if done:
            reward = -100

        self.state_pool.append(self.state)
        self.action_pool.append(float(action))
        self.reward_pool.append(reward)

        self.state = next_state
        self.state = torch.from_numpy(self.state).float()
        self.state = Variable(self.state)

        self.steps += 1

        self.t += 1

        if done:
            self.episode_durations.append(self.t + 1)
            self.plot_durations()
            return False
        else:
            return True
    
    def UpdatePolicy(self, gamma):
        # Discount reward
        running_add = 0
        for i in reversed(range(self.steps)):
            if self.reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * gamma + self.reward_pool[i]
                self.reward_pool[i] = running_add

        # Normalize reward
        reward_mean = np.mean(self.reward_pool)
        reward_std = np.std(self.reward_pool)
        for i in range(self.steps):
            self.reward_pool[i] = (self.reward_pool[i] - reward_mean) / reward_std

        # Gradient Desent
        self.optimizer.zero_grad()

        for i in range(self.steps):
            state = self.state_pool[i]
            action = Variable(torch.FloatTensor([self.action_pool[i]]))
            reward = self.reward_pool[i]

            probs = self.policy_net(state)
            m = Bernoulli(probs)
            loss = -m.log_prob(action) * reward  # Negtive score function x reward
            loss.backward()

        self.optimizer.step()

        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []
        self.steps = 0

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(self.episode_durations)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated