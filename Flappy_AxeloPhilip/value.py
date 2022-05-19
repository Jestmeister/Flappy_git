
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



#this shuld take in a state vector and estimate the disconnted reward based on the state
class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()

        self.fc1 = nn.Linear(7, 36)
        self.fc2 = nn.Linear(36, 36)
        self.fc3 = nn.Linear(36, 1)

        #self.fc1 = nn.Linear(6, 60)
        #self.fc2 = nn.Linear(60, 60)
        #self.m = nn.MaxPool1d(6, stride=4)
        #self.fc3 = nn.Linear(14, 14)
        #self.fc4 = nn.Linear(14, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.m(x)
        x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        
        return x



class Value:
    def __init__(self, learning_rate):
        self.valueNet = ValueNet() #create a net obj
        self.criterion = nn.MSELoss() #create function ref
        self.optimizer = torch.optim.RMSprop(self.valueNet.parameters(), lr=learning_rate) #create a optimizer obj

    def GetValue(self, state):
        return self.valueNet(state) #return value estimate based on the state

    def UpdateValueNet(self, rewardTarget, state):
        y = self.valueNet(state) #takes a "list" of states as a tensor obj and returns a "list" outputs

        loss = self.criterion(rewardTarget, y) #calc the loss
        print('Total value loss for this batch: {}'.format(loss.item()))
        print('')

        loss.backward() #determines the gradient

        self.optimizer.step() #learn

        self.valueNet.zero_grad() #reset gradients