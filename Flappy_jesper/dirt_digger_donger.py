import random
import torch
from agent import DQNagent
from agent_cn import DQNagent_cn


run_counter = 0
go_main = False
if go_main:
    while True:
        run_counter += 1
        print('')
        print('----------- NEW RUN -----------')
        print('')
        agent = DQNagent(n_episodes=750,start_difficulty=4)  #make 0
        agent.train()
        if agent.difficulty == 4 and agent.best_score > 5:
            print(f'Number of runs until epicness: {run_counter}')
            break   
else:
    #CNagent=DQNagent_cn(11,0)
    #CNagent.test_scrn()
    #CNagent.train()
    while True:
        run_counter += 1
        print('')
        print('----------- NEW RUN -----------')
        print('')
        agent = DQNagent(n_episodes=500,start_difficulty=4)  #make 0
        agent.train()
        if agent.difficulty == 4 and agent.best_score > 5:
            print(f'Number of runs until epicness: {run_counter}')
            break   
    

