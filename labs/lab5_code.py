import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from collections import deque  # for memory
from tqdm import tqdm          # for progress bar

from multiprocessing.sharedctypes import Value
#from random import random
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
        x = torch.tensor(x, dtype=torch.float32)
        x = self.forward(x)
        return torch.argmax(x)
        #x = torch.argmax(x, dim=0)
        # return x.detach().numpy().item()


#env = gym.make('CartPole-v1')


class Agent:
    def __init__(self, observation_size, action_size, memorySize=1000):
        self.observation_size = observation_size
        self.action_size = action_size
        self.model = Model(observation_size, action_size)
        self.discount_value = 0.9
        self.epsilon = 1.0
        self.epsilonDecay = 0.99
        self.epsilonMin = 0.00
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.memorySize = memorySize
        self.memory = deque([], maxlen=self.memorySize)

    def remember(self, state, action, reward, next_state, done):
        #s, a, r, s1, d = transition
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, hasRandom=False):
        if random.uniform(0, 1) < self.epsilon and hasRandom:
            action = env.action_space.sample()
        else:
            action = self.model.Q_predict(state)

        #transition = env.step(action)
        # self.remember(transition)

        #state = torch.tensor(state, dtype=torch.float32)

        # return self.model.predict(state)
        #action = self.model.Q_predict(state)
        return int(action)

    # update model based on replay memory
    # you might want to make a self.train() helper method
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        self.optimizer.zero_grad()  # clean
        for element in minibatch:
            self.train(element)

        self.epsilon *= self.epsilonDecay
        self.optimizer.step()

    def train(self, transition):
        s0, a, r, s1, d = transition
        s0 = torch.tensor(transition[0], dtype=torch.float32)
        a = torch.tensor(transition[1])
        r = torch.tensor(transition[2])
        s1 = torch.tensor(transition[3], dtype=torch.float32)
        d = torch.tensor(transition[4])

        #pred = self.model.Q_predict(torch.tensor(s0))
        if not d:
            #value = self.discount_value * torch.argmax(self.act(s0))
            #value = self.discount_value * self.model.predict(s1)
            #s1 = torch.tensor(s1)
            value = r + self.discount_value * \
                float(torch.max(self.model.forward(s1)))
        else:
            value = r
        #pred = Variable(torch.tensor(pred), requires_grad=True)
        pred = self.model.forward(s0)[a]
        #reward = Variable(torch.tensor(value))

        loss = self.criterion(pred, value)
        loss.backward()
        # self.optimizer.step()


def train(env, agent, fileName, episodes=1000, batch_size=64):  # train for many games
    highestReward = 0
    for e in tqdm(range(episodes)):
        state, _ = env.reset()
        done = False
        totalReward = 0
        while not done:
            # 1. make a move in game.
            action = agent.act(state, True)
            next_state, reward, done, _, _, = env.step(action)
            totalReward += reward
            # 2. have the agent remember stuff.
            agent.remember(state, action, reward, next_state, done)
            # 3. update state
            state = next_state
            # 4. if we have enough experiences in our memory, learn from a batch with replay.
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)

        if totalReward > highestReward:
            highestReward = totalReward
        print("ep: " + str(e) + " reward = " + str(totalReward) +
              " / " + str(highestReward) + " (highest)  |  current_epsilon = " + str(agent.epsilon))
        if e > 0 and e % 10 == 0:
            torch.save(agent.model.state_dict(), fileName+"_"+str(episodes)+".pth")#bruh
            # torch.save(agent.)
            print("¤ q-model saved ¤")
    env.close()


def playtest(env, agent, episodes=10):
    #agent.model.state_dict = torch.load('model300.pth')

    # print(agent.model.state_dict)

    #env = gym.make('CartPole-v1', render_mode='human')

    for e in tqdm(range(episodes)):
        state, _ = env.reset()
        done = False
        totalReward = 0
        while not done:
            action = agent.act(state)
            # print(action)
            state, reward, done, _, _ = env.step(action)
            totalReward += reward

        print("ep: " + str(e) + " reward = " + str(totalReward))

    env.close()


env = gym.make('CartPole-v1', render_mode='human')  # , render_mode='human')
#env = gym.make('CartPole-v1')
agent = Agent(env.observation_space.shape[0], env.action_space.n, 5000)

modelFile = 'model4-memory5k-ep300.pth_300end.pth'

agent.model.load_state_dict(torch.load(modelFile))

#train(env, agent, modelFile, 300)
#torch.save(agent.model.state_dict(), modelFile+"_300end.pth")

playtest(env, agent)
