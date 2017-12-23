import math
import numpy as np

class Gridworld(object):
    def __init__(self,size):

        self.size = size
        self.state = None
        self.min_state = np.zeros(shape=(size*size,1),dtype=np.float32)
        self.max_state = np.ones(shape=(size*size,1),dtype=np.float32)
        self.normalised_state = None
        self.timestep = 0
        self.end_episode = False

    def initialise(self):
        self.state = np.array([0,0])
        self.normalised_state = self.normalise(self.state[0], self.state[1])

        self.timestep = 0
        self.end_episode = False

    def normalise(self,x,y):

        pos = x*self.size + y
        internal_state = np.zeros(shape=(self.size*self.size,1),dtype=np.float32)
        internal_state[pos] = 1.0

        range = self.max_state - self.min_state

        return (internal_state - self.min_state)/range

        # Action can be 0 (move left ) , 1( stay) , 2 ( move forward)
    def Take_Action(self,action):

        x = self.state[0]
        y = self.state[1]

        if action == 0:
            y = y - 1
        elif action == 1:
            y = y + 1
        elif action == 2:
            x = x - 1
        elif action == 3:
            x = x + 1

        x = min(x,self.size-1)
        x = max(x, 0)

        y = min(y, self.size - 1)
        y = max(y, 0)

        self.state = np.array([x,y])

        self.normalised_state = self.normalise(self.state[0], self.state[1])

        self.timestep +=1

    def Get_Reward(self):

        x = self.state[0]
        y = self.state[1]

        # After taking the action, if the state reaches the end terminal state, we will end the episode
        if ( x == self.size - 1  and y == self.size - 1) or self.timestep >= 1000:

            self.end_episode = True

        return -1




