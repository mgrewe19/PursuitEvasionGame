#tensorboard --logdir ./MLP_SpacecraftTasking_tensorboard

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
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
import abc

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose = 1):
        super(SaveOnBestTrainingRewardCallback,self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode {:.2f}".format(self.best_mean_reward, mean_reward))
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self. model.save(self.save_path)
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          print(y)
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

if __name__ == "__main__":
    
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    enviornment = SpaceCraftTaskingEnviornment.Enviornment() #Initalizes the enviornment
    enviornment = Monitor(enviornment, log_dir)
    r = 1.12639*1.4959787e11
    mu = 1.327124e20 #(m^3/s^2)mu of the sun
    v = m.sqrt(mu/r)
    inital_conditions = np.array([0, r, 0, -v, 0, 0, 1000, 1000, 1000, 0, 0, 0, np.linalg.norm(np.array([500, 500, 500]) - np.array([1000, 1000, 1000]))])
     #Bennu orbiting around the sun
        #(0-5 Bennu state vector, 6-11 spacecraft state vectore)
    #enviornment.reset() #Resets the enviornment

    #Checking the Enviornemnt
    env = SpaceCraftTaskingEnviornment.Enviornment()
    # It will check your custom environment and output additional warnings if needed
    check_env(env, warn=True)

    ## Eval Call back
    eval_callback = EvalCallback(enviornment, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)
    ##

    ## Checkpoint Callback
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                         name_prefix='rl_model')
    ##

    n_sampled_goal = 4
    timeSteps = 100000

    possible_actions = np.array([[.1,0,0], [-.1,0,0], [0, .1, 0], [0, -.1, 0], [0, 0, .1], [0, 0, -.1], [0,0,0]])
    action_space = (possible_actions)

    #model = DQN(MlpPolicy,enviornment,exploration_initial_eps=0.1, gradient_steps=-1, gamma=0.95, verbose = 2, policy_kwargs=dict(net_arch = [256, 256, 256]))

          #seed=None, double_q=True, target_network_update_freq=500, prioritized_replay=False,
          #prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
          # prioritized_replay_eps=1e-06, param_noise=False, n_cpu_tf_sess=None,full_tensorboard_log=False)
    model = DQN(MlpPolicy, enviornment, gamma=0.95, learning_rate = 0.05, buffer_size = 1000,
    exploration_fraction=0.75, exploration_final_eps=0.02, exploration_initial_eps=0.3, verbose= 2,
    policy_kwargs=dict(net_arch = [256, 256, 256]), train_freq = 1000, batch_size = 1000, learning_starts = 1000,
    _init_setup_model=True, gradient_steps =-1, tensorboard_log="./MLP_SpacecraftTasking_tensorboard/")

    #To visualize the tensor board tensorboard --logdir ./MLP_SpacecraftTasking_tensorboard

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    model.learn(total_timesteps=timeSteps, log_interval=4, callback = callback)
    model.save("deepq_SpacecraftOrbit")

    temp = stable_baselines3.common.evaluation.evaluate_policy(model, enviornment, n_eval_episodes = 10, deterministic=True, render=False, 
     callback= None, reward_threshold=None, return_episode_rewards=True, warn=True)
     #evaluates the policy 10 times and tells me what the reward is each time

    print(temp)

    print("Done with Training")

    ax = plt.axes(projection='3d')

    obs = np.zeros((13,5000))
    reward = np.zeros((5000))
    obs[:,0] = enviornment.reset() #Resets the enviornment
    i=0

    totalReward = 0

    while i < 5000:
        action, _states = model.predict(obs[:,i], deterministic=False) #deterministic was true
        #print(action)
        obs[:,i], reward[i], done, info = enviornment.step(action)
        #print(reward[i])
        totalReward = totalReward + reward[i]
        i += 1

        #print(reward)
        if done == True:
            print("The total reward that the spacecraft recieved is: ",totalReward)
            #print(i)
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

    results_plotter.plot_results([log_dir], timeSteps, results_plotter.X_TIMESTEPS, "Time varying Reward for Spacecraft attempting to Reach Goal Around Bennu")
    plt.show()

    ax = plt.axes(projection='3d')
    ax.plot(obs[0,0:tempEnd], obs[1,0:tempEnd], obs[2,0:tempEnd], color="red", label="Bennu's Orbit")
    ax.plot(obs[0,0:tempEnd] + 500, obs[1,0:tempEnd] + 500, obs[2,0:tempEnd] + 500, color="blue", label="Goal State")
    ax.plot(obs[0,0:tempEnd] + obs[6,0:tempEnd], obs[1,0:tempEnd] + obs[7,0:tempEnd], obs[2,0:tempEnd] + obs[8,0:tempEnd], color="green", label="Spacecraft around Bennu")
    #ax.scatter(obs[0,0], obs[1,0], obs[2,0], color="red", label="Bennu's Inital Position")
    #ax.scatter(obs[0,0] + 500, obs[1,0] + 500, obs[2,0] + 500, color="blue", label="Inital Goal State")
    #ax.scatter(obs[0,0] + obs[6,0], obs[1,0] + obs[7,0], obs[2,0] + obs[8,0], color="green", label="Spacecraft Inital Position")
    ax.set_xlabel("X location [m]")
    ax.set_ylabel("Y location [m]")
    ax.set_zlabel("Z location [m]")
    ax.legend(loc = "best")
    plt.show()

    fig, ax2 = plt.subplots()
    ax2.plot(np.arange(1,tempEnd+1),reward[0:tempEnd])
    ax2.set(xlabel='Number of Time Steps', ylabel='Reward',
       title='Reward vs time')
    ax2.grid()
    plt.show()
