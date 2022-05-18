
import torch
import torch.nn as nn
import torch.nn.functional as F

import agentPG

import environmentCNN

import valueCNN



class PolicyNetCNN(nn.Module):
    def __init__(self, h, w):
        super(PolicyNetCNN, self).__init__()
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
        outputs = 1
        self.head = nn.Linear(384, outputs)
    
    def forward(self, x):
        x = x[None, :]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 384)
        x = torch.sigmoid(self.head(x))
        return x



class AgentPGCNN(agentPG.AgentPG):
    def StartAgent(self, learning_rate, learning_rate_value, gamma, start_difficulty):
        self.env = environmentCNN.GameCNN() #creates a env obj
        self.value = valueCNN.ValueCNN(learning_rate_value, self.env.scr_height, self.env.scr_width) #creates a value estimation obj
        self.policyNet = PolicyNetCNN(self.env.scr_height, self.env.scr_width) #creates a policy net obj
        self.optimizer = torch.optim.RMSprop(self.policyNet.parameters(), lr=learning_rate) #create a optimizer obj

        self.gamma = gamma #use in DiscountedReward()
        self.start_difficulty = start_difficulty #use in env obj

        #tensors determining the run
        self.state = torch.empty(0, 3, 40, 70, dtype=torch.float32)

        self.reward = torch.empty(0, 1, dtype=torch.float32)
        self.discountedReward = torch.empty(0, 1, dtype=torch.float32)

        self.action = torch.empty(0, 1, dtype=torch.float32)

        self.preTrainValueNet = False

        self.batch_size = 1