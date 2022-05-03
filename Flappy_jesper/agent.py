#import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import pygame

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from environment import environment

#https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_inputs, n_actions, n_episodes):
        super(DQN, self).__init__()
        self.fc1    = nn.Linear(n_inputs, 64)  #change to len(self.output)
        self.fc2    = nn.Linear(64, 64)
        self.fc3    = nn.Linear(64, n_actions) #len(actions)

        self.n_episodes = n_episodes
        self.cur_episode = 0
    

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)    #x = data.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    #Import some classez
    def get_input(self,NNInput):
        self.x_to_pipe = NNInput[0]
        self.p_y = NNInput[1]
        self.y_of_pipe = NNInput[2]
        self.p_vx = NNInput[3]

class DQNagent:
    def __init__(self):   
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10

        # Get screen size so that we can initialize layers correctly based on shape
        # returned from AI gym. Typical dimensions at this point are close to 3x40x90
        # which is the result of a clamped and down-scaled render buffer in get_screen()

        #init_screen = get_screen()
        #_, _, screen_height, screen_width = init_screen.shape

        # Get number of actions from gym action space

        self.n_actions = 2

        self.policy_net = DQN(4, self.n_actions).to(device)
        self.target_net = DQN(4, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.episode_durations = []
        self.game = environment(289,511,52,320,34,24,112)
        self.n_episodes = 100


    def select_action(self,state):
        #global steps_done
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)


    

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()



    def train(self):
        #let main loop be in main
        self.cur_episode += 1

        #if isGameOver:

        #update rewards
        #if game over

        if self.cur_episode>self.n_episodes:
            raise SystemExit(0) 
        
        # Initialize the environment and state
        #env.reset()
        #last_screen = get_screen()
        #current_screen = get_screen()
        #state = current_screen - last_screen
        #state = input
        old_state = input
        #for t in count():
        for chunga in range(self.n_episodes):
            while not self.game.isGameOver:
                # Select and perform an action
                action = self.select_action(state)
                #_, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)

                # Observe new state
                #Send action to board and get new state
                #define a enivornment and use this as a training method instead???

                '''
                last_screen = current_screen
                current_screen = get_screen()
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None
                '''
                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    plot_durations()
                    break
                    # Update the target network, copying all weights and biases in DQN
                    if i_episode % TARGET_UPDATE == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

        print('Complete')