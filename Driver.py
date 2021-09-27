import PursuitEvasionEnviornment #Connects the enviornment file (class)
import PursuitEvasionSimulator #Connects the Simulator file (class)
import numpy as np #This is the package for using the array
import matplotlib.pyplot as plt #This is the package for plotting
import timeit as timer #This is the package for timing how long the code is runnign

name = "__main__" #What are we looking at name for?

if name == "__main__": #What are we using this for?
    enviornment = PursuitEvasionEnviornment.Enviornment() #Initalizes the enviornment
    enviornment.reset() #Resets the enviornment
    Steps = 45 #Number of times to run through the enviornment

    histArray = np.zeros(Steps) #Preallocates an array to hold the history data

    start = timer.default_timer() #Gets the start time

    for i in range(Steps): #For loop to run the enviornment
        histArray[i] = enviornment.step() #Saves the history data from the enviornment
    States.enviornment.states.state_history #I don't know what this line is supposed to do

    end = timer.default_timer() #Gets the end time

    totTime = end - start #Gets the total time that passes
    timeArr = np.arange(0, totTime, Steps) #Converts the time that passed into an array

    plt.plot(timeArr, histArray, marker = 'o') #Plots the time vs history
    plt.show() #Shows the plot