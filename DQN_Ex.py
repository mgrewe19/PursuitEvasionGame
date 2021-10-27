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
#from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import MlpPolicy

if __name__ == "__main__":
    enviornment = SpaceCraftTaskingEnviornment.Enviornment() #Initalizes the enviornment
    r = 1.12639*1.4959787e11
    mu = 1.327124e20 #(m^3/s^2)mu of the sun
    v = m.sqrt(mu/r)
    inital_conditions = np.array([0, r, 0, -v, 0, 0, 1000, 1000, 1000, 0, 0, 0]) #Bennu orbiting around the sun
        #(0-5 Bennu state vector, 6-11 spacecraft state vectore)
    #enviornment.reset(inital_conditions) #Resets the enviornment

    #Checking the Enviornemnt
    env = SpaceCraftTaskingEnviornment.Enviornment()
    # It will check your custom environment and output additional warnings if needed
    check_env(env, warn=True)

    ###################################### This is where Im trying to pass in the action space
    possible_actions = np.array([[1,0,0], [-1,0,0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0,0,0]])
    action_space = (possible_actions)

    model = DQN(MlpPolicy, enviornment, verbose=1, exploration_initial_eps=0.1)
    model.learn(total_timesteps=100000, log_interval=4)
    model.save("deepq_SpacecraftOrbit")

    print("Done with Training")

    del model # remove to demonstrate saving and loading

    model = DQN.load("deepq_SpacecraftOrbit")

    #obs = enviornment.reset(inital_conditions) #Resets the enviornment

    ax = plt.axes(projection='3d')

    obs = np.zeros((12,5000))
    obs[:,0] = enviornment.reset(inital_conditions) #Resets the enviornment
    i=0

    while i < 5000:
        action, _states = model.predict(obs[:,i])#, deterministic=True)
        #print(action)
        obs[:,i], reward, done, info = enviornment.step(action)
        i += 1

        #print(reward)
        if done:
            tempEnd = i
            break   

    ######################################

    ax = plt.axes(projection='3d')
    ax.plot3D(obs[0,0:tempEnd], obs[1,0:tempEnd], obs[2,0:tempEnd], color="red", label="Bennu's Orbit")
    ax.plot3D(obs[0,0:tempEnd] + 500, obs[1,0:tempEnd] + 500, obs[2,0:tempEnd] + 500, color="blue", label="Goal State")
    ax.plot3D(obs[0,0:tempEnd] - obs[6,0:tempEnd], obs[1,0:tempEnd] - obs[7,0:tempEnd], obs[2,0:tempEnd] - obs[8,0:tempEnd], color="green", label="Spacecraft around Bennu")
    ax.scatter(obs[0,0], obs[1,0], obs[2,0], color="red", label="Bennu's Inital Position")
    ax.scatter(obs[0,0] + 500, obs[1,0] + 500, obs[2,0] + 500, color="blue", label="Inital Goal State")
    ax.scatter(obs[0,0] - obs[6,0], obs[1,0] - obs[7,0], obs[2,0] - obs[8,0], color="green", label="Spacecraft Inital Position")
    #ax.set_xlabel("X location [m]")
    #ax.set_ylabel("Y location [m]")
    #ax.set_zlabel("Z location [m]")
    #ax.legend(loc = "best")

    plt.show()
