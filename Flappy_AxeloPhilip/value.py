
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



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



class Value:
    def __init__(self, learning_rate):
        self.valueNet = ValueNet()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.valueNet.parameters(), lr=learning_rate)

    def GetValue(self, state):
        return self.valueNet(state)

    def UpdateValueNet(self, rewardTarget, state):
        y = []
        for i in range(len(state)):
            y.append(self.valueNet(state[i]))
        
        y = torch.Tensor(y)
        y = Variable(y.data, requires_grad=True)
        rewardTarget = torch.Tensor(rewardTarget)
        rewardTarget = Variable(rewardTarget.data, requires_grad=True)
        print(y.requires_grad)
        print(rewardTarget.requires_grad)
        loss = self.criterion(rewardTarget, y)

        loss.backward()

        self.optimizer.step()

        self.valueNet.zero_grad()