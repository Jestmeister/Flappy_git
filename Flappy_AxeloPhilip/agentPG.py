
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

        self.fc1 = nn.Linear(7, 36)
        self.fc2 = nn.Linear(36, 36)
        self.fc3 = nn.Linear(36, 1)

        #self.fc1 = nn.Linear(6, 60)
        #self.fc2 = nn.Linear(60, 60)
        #self.m = nn.MaxPool1d(6, stride=4)
        #self.fc3 = nn.Linear(14, 14)
        #self.fc4 = nn.Linear(14, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        #if len(x):x = x[None, :]
        #x = self.m(x)
        x = torch.sigmoid(self.fc3(x))
        #x = torch.sigmoid(self.fc4(x))
        
        return x



class AgentPG:
    def StartAgent(self, learning_rate, learning_rate_value, gamma, start_difficulty):
        self.env = environment.Game() #creates a env obj
        self.value = value.Value(learning_rate_value) #creates a value estimation obj
        self.policyNet = PolicyNet() #creates a policy net obj
        self.policyNetOld = self.policyNet
        self.fMSELoss = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.policyNet.parameters(), lr=learning_rate) #create a optimizer obj

        self.gamma = gamma #use in DiscountedReward()
        self.start_difficulty = start_difficulty #use in env obj

        #tensors determining the run
        self.state = torch.empty(0, 7, dtype=torch.float32)
        
        self.reward = torch.empty(0, 1, dtype=torch.float32)
        self.discountedReward = torch.empty(0, 1, dtype=torch.float32)
        
        self.action = torch.empty(0, 1, dtype=torch.float32)

        self.preTrainValueNet = True

        self.batch_size = 1

    def StartEnv(self):
        self.env.Start(not(self.preTrainValueNet) or False, False, self.start_difficulty) #start game
        startState, not_used, not_used = self.env.Update(False)
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
        #random jump if pre training
        if self.preTrainValueNet:
            return random.random() < 0.1

        actionProbabilityDis = self.policyNet(currentState) #takes a state and returns policy dis

        #adds policy % to "list"
        actionProbability = actionProbabilityDis.view(1, 1)
        self.action = torch.cat((self.action, actionProbability), 0)

        #selects random action based on policy
        selection = random.random()
        if selection < actionProbabilityDis.item():
            return True
        else:
            return False
    


    def UpdatePolicy(self):
        print('Tot discountedReward per run this batch: {}'.format(torch.sum(self.discountedReward).item() / self.batch_size))
        print('Mean discountedReward per state this batch: {}'.format(torch.mean(self.discountedReward).item()))
        
        self.DiscountedReward() #calc DiscountedReward for last game of batch
        
        #update PolicyNet if not value net pre training
        if not(self.preTrainValueNet):
            loss = self.Loss(self.action)
            
            loss.backward()

            self.optimizer.step()

            self.policyNet.zero_grad()

        self.value.UpdateValueNet(self.discountedReward, self.state) #update value net
    
    def ResetParm(self):
        #resets the (training) data
        self.state = torch.empty(0, 7, dtype=torch.float32)

        self.discountedReward = torch.empty(0, 1, dtype=torch.float32)

        self.action = torch.empty(0, 1, dtype=torch.float32)

    #calculates the discounted reward from reward and appends it to discountedReward
    def DiscountedReward(self):
        for i in range(len(self.reward)): #run through all rewards
            currentDiscountedReward = 0
            for j in range(len(self.reward) - i): #run through all future rewards and apply gamma
                currentDiscountedReward += (self.gamma**j) * self.reward[j + i].item()

            currentDiscountedRewardTensor = torch.tensor([[currentDiscountedReward]], dtype=torch.float32) #convert to tensor
            self.discountedReward = torch.cat((self.discountedReward, currentDiscountedRewardTensor), 0) #add to "list"
        
        self.reward = torch.empty(0, 1, dtype=torch.float32) #reset reward "list"

    def Loss(self, actions):
        A = self.discountedReward - self.value.GetValue(self.state) #calc advantage
        #A = A.detach()

        #E[log(policy)A]
        loss = torch.log(actions) * A #loss
        loss = torch.mean(loss) #expected

        print('Total loss for this batch: {}'.format(loss.item()))

        return loss
