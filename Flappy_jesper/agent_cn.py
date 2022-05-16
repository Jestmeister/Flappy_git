#import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from PIL import Image
import pygame

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

#TODO: Save weights and read in weights

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


class DQN_cn(nn.Module):

    def __init__(self,  h, w, outputs):
        super(DQN_cn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)




        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        self.linear_input_size = convw * convh * 32
        #self.head = nn.Linear(self.linear_input_size, outputs)
        self.head = nn.Linear(384, outputs)
   
    def forward(self, x):
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 384)
        x = self.head(x)
        return x
        
 



class DQNagent_cn:
    def __init__(self, n_episodes, start_difficulty):   
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.95
        self.EPS_END = 0.001
        self.EPS_DECAY = 60
        self.TARGET_UPDATE = 20

        self.n_episodes = n_episodes
        self.difficulty = start_difficulty

        self.scr_width = 289
        self.scr_height = 511
        self.play_ground = self.scr_height * 0.8
        self.game = environment(self.scr_width,self.scr_height,52,320,34,24,112,difficulty = self.difficulty)
        

        self.action = 0

        self.n_actions = 2
        self.n_hidden = 16
        self.n_input = len(self.game.cur_state)

        self.policy_net = DQN_cn(self.scr_height, self.scr_width, self.n_actions).to(device)
        self.target_net = DQN_cn(self.scr_height, self.scr_width, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.best_score = 0
        
        self.reward_ls = np.zeros(self.n_episodes)
        self.resize = T.Compose([T.ToPILImage(),
            T.Resize(40, interpolation=Image.CUBIC),
            T.ToTensor()])

        pygame.init()
        self.display_screen_window = pygame.display.set_mode((self.scr_width, self.scr_height))
        self.game_image = {}
        player = 'images/bird.png'
        bcg_image = 'images/background.png'
        pipe_image = 'images/pipe.png'
        self.game_image['numbers'] = (
        pygame.image.load('images/0.png').convert_alpha(),
        pygame.image.load('images/1.png').convert_alpha(),
        pygame.image.load('images/2.png').convert_alpha(),
        pygame.image.load('images/3.png').convert_alpha(),
        pygame.image.load('images/4.png').convert_alpha(),
        pygame.image.load('images/5.png').convert_alpha(),
        pygame.image.load('images/6.png').convert_alpha(),
        pygame.image.load('images/7.png').convert_alpha(),
        pygame.image.load('images/8.png').convert_alpha(),
        pygame.image.load('images/9.png').convert_alpha(),
    )

        self.game_image['message'] = pygame.image.load('images/message.png').convert_alpha()
        self.game_image['base'] = pygame.image.load('images/base.png').convert_alpha()
        self.game_image['pipe'] = (pygame.transform.rotate(pygame.image.load(pipe_image).convert_alpha(), 180),
                            pygame.image.load(pipe_image).convert_alpha()
                            )

        self.game_image['background'] = pygame.image.load(bcg_image).convert()
        self.game_image['player'] = pygame.image.load(player).convert_alpha()
    

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.render().transpose((2, 0, 1))
        
        
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0)


    def test_scrn(self):
        plt.figure()
        plt.imshow(self.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
                interpolation='none')
        plt.title('Example extracted screen')
        plt.show()

    def select_action(self):
        #global steps_done
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        
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
            #monkey = [False, True]
            #return random.randint(0,1)
            self.action = random.randint(0,1)
        else:
            #print('Policy action')
            #Argmax
            state = self.state#.to(dtype=torch.double)
            
            self.action  = self.target_net(state).argmax().item()  #Dont give optimize a boolean?
            #self.action = self.policy_net(state).max(1)[1].view(1, 1)
            #print(self.action)
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
        #print(self.policy_net(state_batch).shape)
        #print(action_batch.shape)
        #print(action_batch.reshape(action_batch.size()[0],1).shape)
        #print(self.state.shape)

        #state_action_values = self.policy_net(state_batch).gather(1, action_batch.reshape(action_batch.size()[0],1))
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

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


    #Implement read state
    def train(self):
        ramp_up = False
        for cur_episode in range(self.n_episodes):
            if (cur_episode+1) % 10 == 0:    
                print(cur_episode+1)
            if ramp_up:
                print('Rampin that booty up!')
                ramp_up = False
                self.difficulty += 1
                #Skip diff 1
                if self.difficulty == 1:
                    self.difficulty += 1
                self.best_score = 0
                self.game = environment(289,511,52,320,34,24,112,difficulty = self.difficulty)
            frames_cleared = 0
            reward = 1
            #term = torch.tensor([1])
            self.game.update(False)
            last_screen = self.get_screen()
            current_screen = self.get_screen()
    
            self.state = current_screen - last_screen
            while not self.game.isGameOver:
                # Select and perform an action
                #print(state)
                old_state = cp.deepcopy(self.state)
                #print(old_state)
                #old_state = torch.tensor([old_state])
                self.select_action()
                self.game.update(self.read_action())
                last_screen = cp.deepcopy(current_screen)
                current_screen = self.get_screen()
                self.state = current_screen - last_screen
                #self.state = cp.deepcopy(torch.tensor(self.state))
                #state = torch.tensor([state])
                
                frames_cleared += 1
                #reward = frames_cleared #+ self.game.score*100
                if self.difficulty == 4 and self.best_score < self.game.score:
                    #torch.save(self.policy_net.state_dict(), 'C:/Users/jespe/Documents/GitHub/Flappy_git/Flappy_jesper/net.pt')
                    torch.save(self.policy_net.state_dict(), 'C:/Users/Jesper/OneDrive/Dokument/GitHub/Flappy_git/Flappy_jesper/net.pt')
                if self.best_score < self.game.score:
                    self.best_score = self.game.score
                if self.game.isGameOver:
                    self.reward_ls[cur_episode] = frames_cleared
                    reward = -100
                    #term = torch.tensor([0])
                if self.game.score == 50:
                    self.reward_ls[cur_episode] = frames_cleared
                    reward = 100
                    ramp_up = True
                
                reward = torch.tensor([reward], device=device)
                action = cp.deepcopy(torch.tensor([[self.action]], dtype=torch.long))

                # Store the transition in memory
                self.memory.push(old_state, action, self.state , reward)

                

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

    def render(self):
        
        self.display_screen_window.blit(self.game_image['background'], (0, 0))
        for pip_upper, pip_lower in zip(self.game.up_pips, self.game.low_pips):
            self.display_screen_window.blit(self.game_image['pipe'][0], (pip_upper['x'], pip_upper['y']))
            self.display_screen_window.blit(self.game_image['pipe'][1], (pip_lower['x'], pip_lower['y']))

        self.display_screen_window.blit(self.game_image['base'], (self.game.b_x, self.play_ground))
        self.display_screen_window.blit(self.game_image['player'], (self.game.p_x, self.game.p_y))
        d = [int(x) for x in list(str(self.game.score))]
        w = 0
        for digit in d:
            w += self.game_image['numbers'][digit].get_width()
        Xoffset = (self.scr_width - w) / 2

        for digit in d:
            self.display_screen_window.blit(self.game_image['numbers'][digit], (Xoffset, self.scr_height * 0.12))
            Xoffset += self.game_image['numbers'][digit].get_width()
        imgdata = pygame.surfarray.array3d(self.display_screen_window) 
        return imgdata
        #pygame.display.update()
        #time_clock.tick(FPS)