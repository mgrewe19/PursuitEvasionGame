import gym
from gym import spaes

class Simulator():
    def __init__(IC = None):
        if IC:
            self.state = IC
        else:
            self.new_IC()
        self.dt = 1
        self.N = 0 #Mean motion (need to choose LEO, GEO, MEO, ...)
    def new_IC():
        self.state = np.array([])
    def propogate_state():
        new_state = np.zeros(6,1)
        k1 = 0 #Runge kutta methods
        k2 = 0
        k3 = 0
        k4 = 0
        self.state += (1/6)*(k1 + 2 * k2 + 2 * k3 + k4)
    def equations_of_motion():
        1+1 #Add in the equations of motions to this part
    def step():
        for i in range(self.StepSize/self.dl):
            self.propogate_state