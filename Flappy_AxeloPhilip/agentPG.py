
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import random

import environment

import value



#this shuld take a state and retunr a probility dis that termines the action
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 36)

        #self.fc3 = nn.Linear(36, 36)
        #self.fc4 = nn.Linear(36, 36)

        self.fc5 = nn.Linear(36, 24)
        self.fc6 = nn.Linear(24, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        
        return x



class AgentPG:
    def StartAgent(self, learning_rate, learning_rate_value, gamma, start_difficulty):
        self.env = environment.Game() #creates a env obj
        self.value = value.Value(learning_rate_value) #creates a value estimation obj
        self.policyNet = PolicyNet() #creates a policy net obj
        self.optimizer = torch.optim.RMSprop(self.policyNet.parameters(), lr=learning_rate) #create a optimizer obj

        self.gamma = gamma #use in DiscountedReward()
        self.start_difficulty = start_difficulty #use in env obj

        #tensors determining the run
        self.state = torch.empty(0, 4, dtype=torch.float32)
        
        self.reward = torch.empty(0, 1, dtype=torch.float32)
        self.discountedReward = torch.empty(0, 1, dtype=torch.float32)

        self.action = torch.empty(0, 1, dtype=torch.float32)

        self.preTrainValueNet = False

    def StartEnv(self):
        startState = self.env.Start(not(self.preTrainValueNet), False, self.start_difficulty) #start game
        self.state = torch.cat((self.state, startState), 0) #append start state to states "list"

        self.DiscountedReward() #calc DiscountedReward from reward + empty reward

    

    #plays one frame of the game and saves the state, reward and action for UpdatePolicy()
    def Update(self):
        currentAction = self.SelectAction(self.state[len(self.state) - 1]) #selects the action based on the last state

        currentState, currentReward, gameover = self.env.Update(currentAction) #runs a frame of the game

        #adds the state to the "list" if not gameover
        if not(gameover):
            self.state = torch.cat((self.state, currentState), 0)

        self.reward = torch.cat((self.reward, currentReward), 0) #adds the reward to the "list"

        return not(gameover) #= running

    #takes the current state as a torch tensor and returns true or false aka jump or not
    def SelectAction(self, currentState):
        if self.preTrainValueNet:
            return random.random() < 0.1

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
        
        if not(self.preTrainValueNet):
            loss = self.Loss(self.action)
            
            loss.backward()

            self.optimizer.step()

            self.policyNet.zero_grad()

        self.value.UpdateValueNet(self.discountedReward, self.state)

        #resets the (training) data
        self.state = torch.empty(0, 4, dtype=torch.float32)

        self.discountedReward = torch.empty(0, 1, dtype=torch.float32)

        self.action = torch.empty(0, 1, dtype=torch.float32)

    #calculates the discounted reward from reward and appends it to discountedReward
    def DiscountedReward(self):
        for i in range(len(self.reward)):
            currentDiscountedReward = 0
            for j in range(len(self.reward) - i):
                currentDiscountedReward += (self.gamma**j) * self.reward[j + i].item()

            currentDiscountedRewardTensor = torch.tensor([[currentDiscountedReward]], dtype=torch.float32)
            self.discountedReward = torch.cat((self.discountedReward, currentDiscountedRewardTensor), 0)
        
        self.reward = torch.empty(0, 1, dtype=torch.float32)

    def Loss(self, actions):
        A = self.discountedReward - self.value.GetValue(self.state)
        A = A.detach()

        #E[log(policy)A]
        loss = torch.log(actions) * A
        loss = torch.mean(loss)

        print('Total loss for this batch: {}'.format(loss.item()))

        return loss
