import random
import torch
from agent import DQNagent

go_main = True
if go_main:
    agent = DQNagent(n_episodes=500)
    agent.train()

'''
#Wrong axis??
a = torch.tensor([[1, 2], [3, 4], [5, 6]])
b = torch.gather(a, 1, torch.tensor([[0], [0], [1]]))

print(b)

to = (torch.tensor([1]),torch.tensor([1]),torch.tensor([0]))
a = torch.cat(to)
#a = torch.reshape(a,(3,1))
print(a.size()[0])
'''

