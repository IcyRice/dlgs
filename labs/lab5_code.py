import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from collections import deque  # for memory
from tqdm import tqdm          # for progress bar

from multiprocessing.sharedctypes import Value
from random import random
from tkinter import Variable

# testrun the gym


def play_no_agent():
    env = gym.make('CartPole-v1', render_mode='human')
    for _ in tqdm(range(10)):
        state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
    env.close()

# play_no_agent()


class Model(nn.Module):
    def __init__(self, observation_size, action_size):
        super(Model, self).__init__()
        # input state-space
        self.dense1 = nn.Linear(observation_size, 100)
        self.dense2 = nn.Linear(100, action_size)
        # output action-space

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x

    def Q_predict(self, x):
        x = self.forward(x)
        x = torch.argmax(x, dim=0)
        return x.detach().numpy().item()


class Agent:
    _
