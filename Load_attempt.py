import SpaceCraftTaskingEnviornment #Connects the enviornment file (class)
import SpaceCraftTaskingSimulator #Connects the Simulator file (class)
import numpy as np #This is the package for using the array
import matplotlib.pyplot as plt #This is the package for plotting
import timeit as timer #This is the package for timing how long the code is runnign
from mpl_toolkits import mplot3d
from stable_baselines3.common.env_checker import check_env
from gym import spaces
import math as m
import gym
import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
import os
from stable_baselines3 import DDPG
#from stable_baselines3.ddpg.policies import LnMlpPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
#from stable_baselines3.common.noise import AdaptiveParamNoiseSpec
from stable_baselines3.common.callbacks import BaseCallback

if __name__ == "__main__":
    enviornment = SpaceCraftTaskingEnviornment.Enviornment() #Initalizes the enviornment
    r = 1.12639*1.4959787e11
    mu = 1.327124e20 #(m^3/s^2)mu of the sun
    v = m.sqrt(mu/r)
    inital_conditions = np.array([0, r, 0, -v, 0, 0, 1000, 1000, 1000, 0, 0, 0, np.linalg.norm(np.array([500, 500, 500]) - np.array([1000, 1000, 1000]))])

    model = DQN(MlpPolicy,enviornment,exploration_initial_eps=0.1, gradient_steps=-1, gamma=0.95, verbose = 1, policy_kwargs=dict(net_arch = [256, 256, 256]))
    model = DQN.load("deepq_SpacecraftOrbit1")

    #obs = enviornment.reset(inital_conditions) #Resets the enviornment

    ax = plt.axes(projection='3d')

    obs = np.zeros((13,5000))
    reward = np.zeros((5000))
    obs[:,0] = enviornment.reset(inital_conditions) #Resets the enviornment
    i=0

    while i < 5000:
        action, _states = model.predict(obs[:,i], deterministic=True)
        #print(action)
        obs[:,i], reward[i], done, info = enviornment.step(action)
        print(reward[i])
        i += 1

        #print(reward)
        if done:
            print("The total reward that the spacecraft recieved is: ",reward[i-1])
            tempEnd = i
            break   

    ######################################

    obs[0:3,:] = obs[0:3,:] * enviornment.Simulator.rCircular
    obs[3:6,:] = obs[3:6,:] * enviornment.Simulator.vCircular
    obs[6:9,:] = obs[6:9,:] * enviornment.Simulator.rSpaceCraft
    obs[9:12,:] = obs[9:12,:] * enviornment.Simulator.vSpaceCraft
    obs[12,:] = obs[12,:] * enviornment.Simulator.distGoal

    for j, nada in enumerate(obs):
        temp = (obs[12,j])
        if(temp <= 200):
            print("The space craft successfully reached the goal")

    ax = plt.axes(projection='3d')
    ax.plot3D(obs[0,0:tempEnd], obs[1,0:tempEnd], obs[2,0:tempEnd], color="red", label="Bennu's Orbit")
    ax.plot3D(obs[0,0:tempEnd] + 500, obs[1,0:tempEnd] + 500, obs[2,0:tempEnd] + 500, color="blue", label="Goal State")
    ax.plot3D(obs[0,0:tempEnd] + obs[6,0:tempEnd], obs[1,0:tempEnd] + obs[7,0:tempEnd], obs[2,0:tempEnd] + obs[8,0:tempEnd], color="green", label="Spacecraft around Bennu")
    ax.scatter(obs[0,0], obs[1,0], obs[2,0], color="red", label="Bennu's Inital Position")
    ax.scatter(obs[0,0] + 500, obs[1,0] + 500, obs[2,0] + 500, color="blue", label="Inital Goal State")
    ax.scatter(obs[0,0] + obs[6,0], obs[1,0] + obs[7,0], obs[2,0] + obs[8,0], color="green", label="Spacecraft Inital Position")
    #ax.set_xlabel("X location [m]")
    #ax.set_ylabel("Y location [m]")
    #ax.set_zlabel("Z location [m]")
    #ax.legend(loc = "best")

    plt.show()