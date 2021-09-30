import gym
from gym import spaces
import numpy as np
import PursuitEvasionSimulator #Connects the Simulator file (class)

class Enviornment(gym.Env):
    #Custom Enviornment that follows the gym interface
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.Simulator = None

    def step(self):
        observation = self.Simulator.step()
        return observation
    def reset(self, IC = None):
        self.Simulator = PursuitEvasionSimulator.Simulator(IC = IC)