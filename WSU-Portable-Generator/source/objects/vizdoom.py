import torch
import random
import math
import gym
from gym import spaces
import numpy as np


def vectorize_object(obj):
    vector = []
    for key in obj:
        if key == 'id' or type(obj[key]) is str:
            continue
        vector.append(obj[key])
    return np.array(vector)


def vectorize_list(lst):
    vector = []
    for obj in lst:
        vector.append(vectorize_object(obj))
    return np.array(vector).reshape(-1)


# assumes there can be a max of four enemies/health/ammo/traps/obstacles
def vectorize_state(state):
    vector = np.zeros(90)
    vector[0:len(state['enemies']) * 5] = vectorize_list(state['enemies'])
    vector[20:20 + len(state['items']['health']) *
           4] = vectorize_list(state['items']['health'])
    vector[36:36 + len(state['items']['ammo']) *
           4] = vectorize_list(state['items']['ammo'])
    vector[52:52 + len(state['items']['trap']) *
           4] = vectorize_list(state['items']['trap'])
    vector[68:68 + len(state['items']['obstacle']) *
           4] = vectorize_list(state['items']['obstacle'])
    vector[84:] = vectorize_object(state['player'])
    return vector

def compute_reward(state, action, next_state):
    player = state['player']
    if len(state['enemies']) > len(next_state['enemies']):
        return 1
    elif action == 'shoot':
        for i in range(len(state['enemies'])):
            enemy = state['enemies'][i]
            enemy_prime = next_state['enemies'][i]
            dy = enemy['y_position'] - player['y_position']
            dx = enemy['x_position'] - player['x_position']
            angle = math.atan2(dy, dx) * 180 / math.pi
            angle += 360 if angle < 0 else 0
            if abs(angle - player['angle']) <= 5 and enemy_prime['health'] < enemy['health']:
                return 0.5
    return 0


class VizDoomEnv(gym.Env):

    def __init__(self):
        super(VizDoomEnv, self).__init__()
        self.states = None
        self.rewards = None
        self.dones = None
        self.current_step = 0
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(-512, 512, shape=(90,))

    def load_trajectory(self, states, rewards, dones):
        self.states = states
        self.rewards = rewards
        self.dones = dones

    def reset(self):
        assert self.states is not None
        state = self.states[0]
        self.current_step = 0
        return state

    def step(self, action):
        assert self.states is not None
        assert self.rewards is not None
        assert self.dones is not None
        next_state = self.states[self.current_step + 1]
        reward = self.rewards[self.current_step]
        done = self.dones[self.current_step]
        self.current_step += 1
        return next_state, reward, done, {}

    def render(self):
        raise NotImplementedError()
