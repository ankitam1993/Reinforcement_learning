import math
import numpy as np
uMax = 5.0      # Maximum torque in either direction
dt = 0.01	    # Time step size
simSteps = 10	# Dynamics simulated at dt/simSteps
m = 1	        # Mass
l = 1 	        # Length
g = 9.8	        # Gravity
mu = 0.1	    # Friction constant

# angle : -pi, pi
# angular vel : -3pi, 3pi

class Pendulum(object):
    def __init__(self,angles,velocities):
        self.angles = angles
        self.velocities = velocities
        self.state = None
        self.normalised_state = None
        self.timestep = 0
        self.end_episode = False

    def initialise(self):
        angle = np.random.uniform(-1*math.pi,math.pi)
        vel = 0.0
        self.state = np.array([angle,self.bound(vel)])
        self.normalised_state = np.array(self.normalise(self.wrapPosNegPI(angle), vel))

        self.timestep = 0
        self.end_episode = False

    def normalise(self,pos,vel):
        norm_pos = (pos - self.angles[0]) / (2*math.pi)
        norm_vel = (vel - self.velocities[0]) / (6*math.pi)

        #print pos, vel
        #print norm_pos , norm_vel
        return norm_pos , norm_vel

    def wrapPosNegPI(self,theta):

        a = theta + math.pi
        b = 2*math.pi

        c = a%b - math.pi

        return c

    def bound(self,omega):

        return min(self.velocities[1],max(self.velocities[0],omega))

    # Action will be according to gaussian distribution
    def Take_Action(self,action):

        theta = self.state[0]
        omega = self.state[1]

        if (action < -uMax):
            action = -uMax
        if (action > uMax):
            action = uMax

        #print action
        subDt = dt / simSteps
        thetaDot = omegaDot = subDt
        tUp = 0

        #print simSteps
        for i in range(0,simSteps):
            thetaDot = omega
            omegaDot = (-mu * omega - m * g * l * math.sin(theta) + action) / (m * l * l)

            theta += subDt * thetaDot
            omega += subDt * omegaDot

            temp = self.wrapPosNegPI(theta)

            if (temp * temp > 9.0 * math.pi * math.pi / 16.0):
                tUp += subDt
                #print 'tup' , tUp

        self.state = np.array([theta,self.bound(omega)])
        self.normalised_state = np.array(self.normalise(self.wrapPosNegPI(self.state[0]), self.state[1]))
        self.timestep += dt

    def Get_Reward(self):

        theta = self.state[0]
        omega = self.state[1]

        # After taking the action, if the state reaches the end terminal state, we will end the episode
        if self.timestep >= 10.0:

            self.end_episode = True

        reward = -1*math.cos(theta) - (omega * omega) / 100.0
        return reward




