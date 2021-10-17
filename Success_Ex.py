import SpaceCraftTaskingEnviornment #Connects the enviornment file (class)
import SpaceCraftTaskingSimulator #Connects the Simulator file (class)
import numpy as np #This is the package for using the array
import matplotlib.pyplot as plt #This is the package for plotting
import timeit as timer #This is the package for timing how long the code is runnign
from mpl_toolkits import mplot3d
from stable_baselines3.common.env_checker import check_env
from gym import spaces
import math as m

if __name__ == "__main__":
    enviornment = SpaceCraftTaskingEnviornment.Enviornment() #Initalizes the enviornment
    r = 1.12639*1.4959787e11
    mu = 1.327124e20 #(m^3/s^2)mu of the sun
    v = m.sqrt(mu/r)
    inital_conditions = np.array([0, r, 0, -v, 0, 0, 1000, 1000, 1000, 0, 0, 0]) #Bennu orbiting around the sun
        #(0-5 Bennu state vector, 6-11 spacecraft state vectore)
    enviornment.reset(inital_conditions) #Resets the enviornment
    Steps = 5000 #Number of times to run through the enviornment

    histArray = np.zeros((12, Steps)) #Preallocates an array to hold the history data

    ###################################### This is where Im trying to pass in the action space
    enviornment.action_space = spaces.Box(low=np.array([-0.1,-0.1,0]),high=np.array([0,0,0]),shape=(3,))

    ######################################

    start = timer.default_timer() #Gets the start time

    for i in range(Steps): #For loop to run the enviornment
        histArray[:,i],_,_,_ = enviornment.step(enviornment.action_space) #Saves the history data from the enviornment
        #print(histArray[:,i])

    end = timer.default_timer() #Gets the end time

    totTime = end - start #Gets the total time that passes
    timeArr = np.linspace(0, totTime, Steps) #Converts the time that passed into an array
    #print(histArray)

    ax = plt.axes(projection='3d')

    ax.plot3D(histArray[0,:], histArray[1,:], histArray[2,:], color="red", label="Bennu's Orbit")
        #Plots the orbit of Bennu about the sun

    ax.plot3D(histArray[0,:] + 500, histArray[1,:] + 500, histArray[2,:] + 500, color="blue", label="Goal State")
        #Plots the goal states position

    ax.plot3D(histArray[0,:] - histArray[6,:], histArray[1,:] - histArray[7,:], histArray[2,:] - histArray[8,:], color="green", label="Spacecraft around Bennu")
        #Plots the orbit of the spacecraft about Bennu

    plt.legend()

    ax.ticklabel_format(useOffset = False)
    plt.show() #Shows the plot