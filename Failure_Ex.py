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
    Steps = 500 #Number of times to run through the enviornment

    ###################################### This is where Im trying to pass in the action space
    possible_actions = np.array([[1,0,0], [-1,0,0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0,0,0]])
    actionNumbers = np.array([5,6,6,6,5,5,5,5,6,4,4,4,4,4,3,3,3,3,3,6,1,1,2,2,2,2,2,1,1,1,2,2])

    histArray = np.zeros((12, len(actionNumbers))) #Preallocates an array to hold the history data

    start = timer.default_timer() #Gets the start time
    #print(actions2take)
    for i, action in enumerate(actionNumbers): #For loop to run the enviornment
        histArray[:,i],_,_,_ = enviornment.step(action) #Saves the history data from the enviornment
        #print(histArray[:,i])
    #print(histArray)
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

    #SC= np.array([histArray[0,temp-1] - histArray[6,temp-1], histArray[1,temp-1] - histArray[7,temp-1], histArray[2,temp-1] - histArray[8,temp-1]])
    #goal = np.array([histArray[0,temp-1], histArray[1,temp-1], histArray[2,temp-1]])
    #print()
    #print(SC)
    #print(goal)
    #temp2 = np.linalg.norm(histArray[6:9,temp-1])
    #print(temp2)
    #print()


    plt.legend()
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    ax.ticklabel_format(useOffset = False)
    plt.show() #Shows the plot