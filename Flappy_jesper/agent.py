#import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from environment import environment
import pandas as pd
import copy as cp

#https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 

#TODO: Normalize inputs

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

    def __init__(self, n_inputs, n_actions, n_hidden):
        super(DQN, self).__init__()
        self.fc1    = nn.Linear(n_inputs, n_hidden)  #change to len(self.output)
        self.fc2    = nn.Linear(n_hidden, n_hidden)
        self.fc3    = nn.Linear(n_hidden, n_actions) #len(actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #x = x.to(device)    #
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
        #qValues = self.myNetwork(x)
        #return torch.sigmoid(qValues)
        



class DQNagent:
    def __init__(self, n_episodes, start_difficulty):   
        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.EPS_START = 0.7
        self.EPS_END = 0.00001
        self.EPS_DECAY = 50
        self.TARGET_UPDATE = 20

        self.n_episodes = n_episodes
        self.difficulty = start_difficulty

        self.game = environment(289,511,52,320,34,24,112,difficulty = self.difficulty)

        self.action = 0

        self.n_actions = 2
        self.n_hidden = 16
        self.n_input = len(self.game.cur_state)

        self.policy_net = DQN(self.n_input, self.n_actions, self.n_hidden).to(device)
        self.target_net = DQN(self.n_input, self.n_actions, self.n_hidden).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr=0.00005)
        self.memory = ReplayMemory(50000)

        self.steps_done = 0
        self.best_score = 0
        
        self.reward_ls = np.zeros(self.n_episodes)



    def select_action(self):
        #global steps_done
        sample = random.random()
        #eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
          #  math.exp(-1. * self.steps_done / self.EPS_DECAY)
        
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * (self.steps_done + self.game.score * 10) / self.EPS_DECAY)
        
        self.steps_done += 1

        #eps_threshold = max(0.0005, 0.7 * math.exp(-1. * self.game.score / self.EPS_DECAY))

        '''
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
        '''
        #monkey = [False, True]
        if sample < eps_threshold:
            #Random
            self.action = random.randint(0,1)
        else:
            #Argmax
            state = torch.tensor(self.game.cur_state)#.to(dtype=torch.double)
            #print(state)
            self.action  = self.target_net(state).argmax().item()  #Dont give optimize a boolean?
        #print(self.action)
            #with torch.no_grad():
            #    self.action = self.policy_net(torch.tensor(self.game.cur_state)).max(1)[1].view(1, 1)

    

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
        #term_batch = torch.cat(batch.term)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
       
        #print(state_batch.shape)
        #print(action_batch.reshape(action_batch.size()[0],1).shape)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.reshape(action_batch.size()[0],1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch  #Add term did nothing...

        # Compute Huber loss
        #criterion = nn.SmoothL1Loss()
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)       #Nummerical clipping the outputs
        self.optimizer.step()

    def read_action(self):
        if self.action == 0:
            return False
        else:
            return True

    #Outputs -1 and 1??? (nope)
    #Set seeds for same pipes all the time??? (Ez mode activated)
    #Select action calls in correct net? (traget)
    #Change network structure? (probs not)
    #Correct inputs??? (Not to many?)

    #Possible improvements:
    #Add punishment for next state being a death? (Did something with expected reward in optimize with term)
    #Epsilon greedy change in epsilon decay?
    #Parameter tuning?
    #Correct reward function?
    #Correct loss function?
    #Correct optimizer?
    #Save best policy run or just check loss function???
    #cnn
    #adaptive epsilon decay (low decay when good and large decay when large)
    #Save correct net!!! Must be a converged net
    #Hard to have empty space in begining and the pipe thight space?

    #Normalize inputs!

    def train(self):
        ramp_up = False
        for cur_episode in range(self.n_episodes):
            if (cur_episode+1) % 10 == 0:    
                print(cur_episode+1)
            if ramp_up:
                ramp_up = False
                self.difficulty += 1
                #Skip diff 1
                if self.difficulty == 1:
                    self.difficulty += 1
                print(f'Difficulty now at {self.difficulty}')
                self.best_score = 0
                self.game = environment(289,511,52,320,34,24,112,difficulty = self.difficulty)
            frames_cleared = 0
            reward = 1
            #term = torch.tensor([1])
            self.game.update(False)
            while not self.game.isGameOver:
                # Select and perform an action
                old_state = cp.deepcopy(torch.tensor([self.game.cur_state]))
                #print(old_state)
                #old_state = torch.tensor([old_state])
                self.select_action()
                
                self.game.update(self.read_action())
                state = cp.deepcopy(torch.tensor([self.game.cur_state]))
                #state = torch.tensor([state])
                
                frames_cleared += 1
                #reward = frames_cleared #+ self.game.score*100
                if self.difficulty == 4 and self.best_score < self.game.score:
                    #torch.save(self.policy_net.state_dict(), 'C:/Users/jespe/Documents/GitHub/Flappy_git/Flappy_jesper/net.pt')
                    torch.save(self.target_net.state_dict(), f'C:/Users/Jesper/OneDrive/Dokument/GitHub/Flappy_git/Flappy_jesper/net{self.game.score}.pt')
                if self.best_score < self.game.score:
                    self.best_score = self.game.score
                if self.game.isGameOver:
                    self.reward_ls[cur_episode] = frames_cleared
                    reward = -1  #-100
                    #term = torch.tensor([0])
                if self.game.score == 30 and self.difficulty < 4:
                    self.reward_ls[cur_episode] = frames_cleared
                    reward = 100
                    ramp_up = True
                
                reward = torch.tensor([reward], device=device)
                action = cp.deepcopy(torch.tensor([self.action], device=device))

                # Store the transition in memory
                self.memory.push(old_state, action, state , reward)

                

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                if ramp_up:
                    #self.game.restart()
                    #self.target_net.load_state_dict(self.policy_net.state_dict())
                    del self.game
                    break
                
        
            # Update the target network, copying all weights and biases in DQN
            if cur_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
        if self.difficulty == 4 and self.best_score > 5:        
            print(f'Achieved difficulty: {self.difficulty}')
            print(f'Best score of run: {self.best_score}')
            #torch.save(self.target_net.state_dict(), 'C:/Users/jespe/Documents/GitHub/Flappy_git/Flappy_jesper/net.pt')

            window_size = 10
            numbers_series = pd.Series(self.reward_ls)
            moving_averages = numbers_series.rolling(window_size).mean()
            moving_averages_list = moving_averages.tolist()
            final_list = moving_averages_list[window_size - 1:]

            plt.plot(self.reward_ls,label = 'Score per ep')
            plt.plot(range(window_size-1,len(self.reward_ls)),final_list,label = f'Score per {window_size} ep')
            plt.legend()
            plt.xlabel('Episode')
            plt.ylabel('Frames cleared')
            plt.show()