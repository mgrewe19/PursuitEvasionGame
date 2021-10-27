import gym
from gym import spaces
import numpy as np
import SpaceCraftTaskingSimulator #Connects the Simulator file (class)
import random
import math

class Enviornment(gym.Env):
    #Custom Enviornment that follows the gym interface
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.Simulator = None
        self.observation_space = spaces.Box(-1e16, 1e16, shape=(12, ))
        self.action_space = spaces.Discrete(7)
    def step(self, action):
        possible_actions = np.array([[1,0,0], [-1,0,0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0,0,0]])
        observation, Done = self.Simulator.step(possible_actions[action])
        return observation, self.Simulator.Current_Reward, Done, {}
    def reset(self, IC = None):
        if IC is not None:
            self.Simulator = SpaceCraftTaskingSimulator.Simulator(IC = IC)
        else:
            r = 1.12639*1.4959787e11
            mu = 1.327124e20 #(m^3/s^2)mu of the sun
            v = -math.sqrt(mu/r)
            #IC = np.array([0, r, 0, -v, 0, 0, 1000, 1000, 1000, 0, 0, 0]) #Bennu orbiting around the sun
            r_hat_asteroid = np.random.rand(1,2)
            r_hat_asteroid = r_hat_asteroid[0]/np.linalg.norm(r_hat_asteroid)
            theta = math.atan2(r_hat_asteroid[1], r_hat_asteroid[0])
            sc_hat = np.random.rand(1,3)
            sc_hat = sc_hat[0]/np.linalg.norm(sc_hat)
            IC = np.array([r*r_hat_asteroid[0], r*r_hat_asteroid[1], 0, v*math.cos(theta), v*math.sin(theta), 0, 3000*sc_hat[0], 3000*sc_hat[1], 3000*sc_hat[2], 0, 0, 0])
            #print(IC[6:9])
            self.Simulator = SpaceCraftTaskingSimulator.Simulator(IC)
        #print(self.Simulator.state.shape)
        return self.Simulator.state