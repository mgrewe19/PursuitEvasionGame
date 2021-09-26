import gym
from gym import spaces
import numpy as np

class Simulator():
    def __init__(IC = None):
        if IC:
            self.state = IC
        else:
            self.new_IC()
        self.dt = 1
        self.N = 11.25 #Mean motion at LEO
    def new_IC():
        self.state = np.array([])
    def propogate_state(self, x_k):
        """Integrates the spacecraft dynamics forward by one time step using the RK4 method """
        #I dont know what x_k is supposed to be

        new_state = np.zeros(6,1)
        k1 = self.dt*self.equations_of_motion(x_k[0:6])
        k2 = self.dt*self.equations_of_motion(x_k[0:6] + 0.5*k1[:,0])
        k3 = self.dt*self.equations_of_motion(x_k[0:6] + 0.5*k2[:,0])
        k4 = self.dt*self.equations_of_motion(x_k[0:6] + k3[:,0])
        self.state += x_k[0:6] + (1.0/6.0)*(k1[:,0] + 2*k2[:,0] + 2*k3[:,0] + k4[:,0]) 

        return self.state

    def equations_of_motion(self, x_k):
        #Im still not sure what x_k is supposed to be doing here
            #Is x_k the equation of state? 1-3 are position, 4-6 are velocity, 7-9 are acceleration
        """Returns the x_dot vector"""
        #Initalizes x
        x_dot = np.zeros((6,1))

        #Velocity
        x_dot[0, 0] = x_k(3)
        x_dot[1, 0] = x_k(4)
        x_dot[2, 0] = x_k(5)

        #Acceleration
        x_dot[3, 0] = 3*x_k[0]*self.n**2 + 2*x_k[4]*self.n
        x_dot[4, 0] = -2*x_k[3]*self.n
        x_dot[5, 0] = -x_k[2]*self.n**2

        return x_dot

    def step():
        for i in range(self.StepSize/self.dt):
            self.propogate_state