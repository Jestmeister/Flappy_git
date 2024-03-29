import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
from torch import nn
from torch import optim

from environment import environment

#https://towardsdatascience.com/breaking-down-richard-suttons-policy-gradient-9768602cb63b

#https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0

class policy_estimator():
    def __init__(self, env):
        self.n_inputs = len(env.cur_state)
        self.n_outputs = 2
        
        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16), 
            nn.ReLU(), 
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1))
    
    def predict(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] 
        for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()

print(discount_rewards([1,2,3,4]))                         #########################################

def read_action(action):
        if action == 0:
            return False
        else:
            return True

def reinforce(env, policy_estimator, num_episodes=2000,
              batch_size=10, gamma=0.99):
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1
    
    # Define optimizer
    optimizer = optim.Adam(policy_estimator.network.parameters(), 
                           lr=0.00005)
    
    action_space = 2
    action_space = np.arange(action_space)
    ep = 0
    while ep < num_episodes:
        env.restart()
        s_0 = env.cur_state
        states = []
        rewards = []
        actions = []
        done = False
        while done == False:
            # Get actions and convert to numpy array
            action_probs = policy_estimator.predict(
                s_0).detach().numpy()
            action = np.random.choice(action_space, 
                p=action_probs)
            
            env.update(read_action(action))
            #s_1, r, done, _ = env.step(action)
            
            s_1 = env.cur_state

            if env.isGameOver:
                r = -100
            else:
                r = 1


            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1
            
            # If done, batch data
            #if done
            if env.isGameOver:
                batch_rewards.extend(discount_rewards(
                    rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))
                
                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(
                        batch_rewards)
                    # Actions are used as indices, must be 
                    # LongTensor
                    action_tensor = torch.LongTensor(
                       batch_actions)
                    
                    # Calculate loss
                    logprob = torch.log(
                        policy_estimator.predict(state_tensor))
                    selected_logprobs = reward_tensor * torch.gather(logprob, 1, 
                        action_tensor).squeeze()
                    loss = -selected_logprobs.mean()
                    
                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1
                    
                avg_rewards = np.mean(total_rewards[-100:])
                # Print running average
                print("\rEp: {} Average of last 100:" +   
                     "{:.2f}".format(
                     ep + 1, avg_rewards), end="")
                ep += 1
                
    return total_rewards

env = environment(289,511,52,320,34,24,112,difficulty = 4)
policy_est = policy_estimator(env)
rewards = reinforce(env, policy_est)

plt.plot(rewards)
plt.show()

#TODO: 
#1. Debug
#2. Save Network
#3. Implement epsilon-greedy???