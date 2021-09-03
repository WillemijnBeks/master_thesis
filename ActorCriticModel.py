#!/usr/bin/env python
# coding: utf-8


import torch
from torch import nn
import numpy as np
eps = np.finfo(np.float32).eps.item()

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, temperature=1):
        super(Actor, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU()
            )

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, action_size), #sum(action_size)
            )

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
            )

        # maybe dim=1 if batchsize is larger than 1
        self.softmax = nn.Softmax(dim=0)
        self.temperature = temperature
        #self.action_size = action_size

    def forward(self, x):
        x = self.main(x)
        action_dist = self.actor_head(x)
        # action_dist = torch.split(action_dist, self.action_size)
        # # This explicitly assumes two actions, change if needed
        # action_dist_1 = self.softmax(action_dist[0]/self.temperature)
        # action_dist_2 = self.softmax(action_dist[1]/self.temperature)

        action_dist = self.softmax(action_dist/self.temperature)
        value = self.critic_head(x)
        #return action_dist_1, action_dist_2, value
        return action_dist, value


class Actor(nn.Module):
    def __init__(self, state_size, action_size , hidden_size, temperature=1):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, sum(action_size))
        self.elu = nn.ELU()
        # maybe dim=1 if batchsize is larger than 1
        self.softmax = nn.Softmax(dim=0)
        self.temperature = temperature
        self.action_size = action_size

    def forward(self, state):
        action_dist = self.elu(self.linear1(state))
        action_dist = self.elu(self.linear2(action_dist))
        action_dist = self.elu(self.linear3(action_dist))
        action_dist = self.linear4(action_dist)

        action_dist = torch.split(action_dist, self.action_size)
        # This explicitly assumes two actions, change if needed
        action_dist_1 = self.softmax(action_dist[0]/self.temperature)
        action_dist_2 = self.softmax(action_dist[1]/self.temperature)       
        return action_dist_1, action_dist_2

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        self.elu = nn.ELU()
        

    def forward(self, state):
        value = self.elu(self.linear1(state))
        value = self.elu(self.linear2(value))
        value = self.elu(self.linear3(value))
        value = self.linear4(value)
        return value


def calc_loss(action_probs, values, rewards, gamma):
    discounted_sum = 0.0
    duration = len(rewards)
    returns = torch.zeros((duration))
    rewards = torch.flip(rewards, [0])

    for i in range(duration):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        returns[i] = discounted_sum

    returns = torch.flip(returns, [0])

    advantage = returns - values.detach()
    
    log_action_probs = torch.log(action_probs)
    loss_actor = torch.mean(-log_action_probs * advantage)
    
    loss_critic = nn.functional.mse_loss(values, returns)

    return loss_actor, loss_critic

