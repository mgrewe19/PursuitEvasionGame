import gym
from gym import spaces
import numpy as np
import SpaceCraftTaskingSimulator #Connects the Simulator file (class)

class Enviornment(gym.Env):
    #Custom Enviornment that follows the gym interface
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.Simulator = None
        self.observation_space = spaces.Box(-1e16, 1e16, shape=(12, ))
        #self.action_space = spaces.Box(-10, 10, shape=(3,1),dtype=np.float32)

    def step(self, action):
        takeAction = action.sample()
        #print(takeAction)
        observation = self.Simulator.step(takeAction)
        return observation, self.Simulator.get_reward(), False, {}
    def reset(self, IC = None):
        self.Simulator = SpaceCraftTaskingSimulator.Simulator(IC = IC)
        #print(self.Simulator.state.shape)
        return self.Simulator.state