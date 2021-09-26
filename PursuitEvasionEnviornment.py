import gym
from gym import spaces
import numpy as np

class Enviornment(gym.Env):
    #Custom Enviornment that follows the gym interface
    metadata = {'render.modes': ['human']}

    def __init__():
        self.Simulator = None

    def step():
        observation = self.Simulator.step()
        return observation
    def reset(self, IC = None):
        self.Simulator = Simulator.Sim(IC = IC)