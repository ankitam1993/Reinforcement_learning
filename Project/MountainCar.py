import math
import numpy as np

class MountainCar(object):
    def __init__(self,start,positions,velocities):

        self.start = start
        self.positions = positions
        self.velocities = velocities
        self.state = None
        self.normalised_state = None
        self.timestep = 0
        self.end_episode = False

    def initialise(self):
        self.state = np.array([self.start[0],self.start[1]])
        self.normalised_state = np.array(self.normalise(self.state[0], self.state[1]))

        self.timestep = 0
        self.end_episode = False

    def normalise(self,pos,vel):
        norm_pos = (pos - self.positions[0]) / 1.8
        norm_vel = (vel - self.velocities[0]) / 0.14
        return norm_pos , norm_vel

    # Action will be according to gaussian distribution
    def Take_Action(self,action):

        cur_pos = self.state[0]
        cur_speed = self.state[1]

        new_speed = cur_speed + 0.001*(action) - 0.0025*math.cos(3*cur_pos)

        if new_speed < self.velocities[0]:
            new_speed = self.velocities[0]

        elif new_speed > self.velocities[-1]:
            new_speed = self.velocities[-1]

        new_pos = cur_pos + new_speed

        # If position reaches the left bound, then we would need to re-initialise the velocity
        if new_pos <= self.positions[0]:
            new_speed = 0

        self.state = np.array([new_pos,new_speed])

        self.normalised_state = np.array(self.normalise(self.state[0],self.state[1]))

        self.timestep +=1

    def Get_Reward(self):

        cur_pos = self.state[0]

        # After taking the action, if the state reaches the end terminal state, we will end the episode
        if cur_pos >= self.positions[-1] or self.timestep == 20000:

            self.end_episode = True
            return 0

        else:
            return -1




