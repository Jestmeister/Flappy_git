import random
import torch
from agent import DQNagent


run_counter = 0
go_main = True
if go_main:
    while True:
        run_counter += 1
        print('')
        print('----------- NEW RUN -----------')
        print('')
        agent = DQNagent(n_episodes=600,start_difficulty=0)
        agent.train()
        if agent.difficulty == 4 and agent.best_score > 5:
            print(f'Number of runs until epicness: {run_counter}')
            break   

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

