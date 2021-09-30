import gym
from gym import spaces
import numpy as np
import PursuitEvasionEnviornment #Connects the enviornment file (class)

class Simulator():
    def __init__(self, IC = None):
        if IC:
            self.state = IC
        else:
            self.new_IC()
        #print(self.state)
        self.dt = 10 #Seconds
        #self.N = 0.0011 #Mean motion at ~500 km
        self.mu = 1.327124e20 #(m^3/s^2)mu of the sun
        self.StepSize = 30 #This initalizes the step size. (In seconds)
    def new_IC(self):
        self.state = np.array([])
    def propogate_state(self, x_k):
        """Integrates the spacecraft dynamics forward by one time step using the RK4 method """
        #print(x_k)

        new_state = np.zeros((12,1))
        
        #print(self.state)
        k1 = self.dt*self.equations_of_motion(x_k)
        #print("k1: ", k1)
        k2 = self.dt*self.equations_of_motion(x_k[0:1] + 0.5*k1[:,0])
        #print("k2: ", k2)
        k3 = self.dt*self.equations_of_motion(x_k[0:1] + 0.5*k2[:,0])
        #print("k3: ", k3)
        k4 = self.dt*self.equations_of_motion(x_k[0:1] + k3[:,0])
        #print("k4: ", k4)
        self.state += (1.0/12.0)*(k1[:,0] + 2*k2[:,0] + 2*k3[:,0] + k4[:,0]) 

        return self.state

    def equations_of_motion(self, x_k):
        #Im still not sure what x_k is supposed to be doing here
            #Is x_k the equation of state? 1-3 are position, 4-6 are velocity, 7-9 are acceleration
        """Returns the x_dot vector"""
        #Initalizes x
        x_dot = np.zeros((12,1))
        #print(x_k)

    #The following information is for Bennu
        #Velocity
        x_dot[0] = x_k[3]
        x_dot[1] = x_k[4]
        x_dot[2] = x_k[5]

        #Acceleration
        r = np.linalg.norm(x_k[0:3])
        x_dot[3] = -self.mu*x_k[0]/(r**3)
        x_dot[4] = -self.mu*x_k[1]/(r**3)
        x_dot[5] = -self.mu*x_k[2]/(r**3)

    #The following information is for the spacecraft orbiting around Bennu
        #Velocity
        x_dot[6] = x_k[9]
        x_dot[7] = x_k[10]
        x_dot[8] = x_k[11] #Should be x_k[11]

        #Acceleration
        r = np.linalg.norm(x_k[6:9])
        x_dot[9] = -self.mu*x_k[6]/(r**3)
        x_dot[10] = -self.mu*x_k[7]/(r**3)
        x_dot[11] = -self.mu*x_k[8]/(r**3)

        return x_dot

    def step(self):
        for i in range(int(self.StepSize/self.dt)):
            self.propogate_state(self.state)
        return self.state