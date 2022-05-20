import random
import torch
from agent import DQNagent
from agent_cn import DQNagent_cn


run_counter = 0
go_main = True
if go_main:
    while True:
        print('dense')
        run_counter += 1
        print('')
        print('----------- NEW RUN -----------')
        print('')
        agent = DQNagent(n_episodes=1000,start_difficulty=4)  
        agent.train()
        if agent.difficulty == 4 and agent.best_score > 5:
            print(f'Number of runs until epicness: {run_counter}')
            break   
else:
    #CNagent=DQNagent_cn(11,0)
    #CNagent.test_scrn()
    #CNagent.train()
    print('cnn')
    while True:
        run_counter += 1
        print('')
        print('----------- NEW RUN -----------')
        print('')
        agent = DQNagent_cn(n_episodes=1000,start_difficulty=4)  #make 0
        agent.train()
        if agent.difficulty == 4 and agent.best_score > 5:
            print(f'Number of runs until epicness: {run_counter}')
            break   
    

