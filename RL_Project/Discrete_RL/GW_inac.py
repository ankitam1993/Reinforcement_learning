from Gridworld import *
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
from joblib import Parallel, delayed
import sys
import os , scipy

size = 5

class GW_INAC(object):

    def __init__(self,episodes,num_trials,lamda,gamma,alpha_actor,alpha_critic,fourier_order):

        self.num_trials = num_trials
        self.num_episodes = episodes
        self.lamda = lamda
        self.gamma = gamma
        self.num_actions = 4
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic

        self.fourier_order = fourier_order
        self.num_states = size*size - 1 # sqaure matrix ( here , 0 - size*size)
        self.features_length =  int(math.pow(self.fourier_order + 1,self.num_states))

    def get_features(self,gw):

        features = np.zeros(shape=(self.features_length, 1), dtype=np.float32)

        c = np.zeros(shape=(1,size*size), dtype=np.float32)
        count = 0

        for i in range(0, self.fourier_order+1):
            for j in range(0, self.fourier_order+1):

                f = c.dot(gw.normalised_state)

                #print i , j , f , math.cos(math.pi *f)
                features[count, 0] = math.cos(math.pi *f)
                count +=1

        return features

    def take_action(self,features,theta):

        action_probablities = self.softmax_policy(features,theta)

        action = np.random.randint(self.num_actions)

        #print action_probablities
        d = np.random.rand()
        sum = 0.0
        for i in range(0,self.num_actions):
            sum += action_probablities[i][0]
            if sum >=d:
                action = i
                break

        #print action
        return action

    # Take action according to softmax policy
    def softmax_policy(self,features,theta):

        action_probabilities = np.ndarray(shape=(self.num_actions,1),dtype=np.float32)

        for i in range(0,self.num_actions):

            action_probabilities[i] = np.dot(features.T,theta[self.features_length*i:self.features_length+self.features_length*i,])[0][0]

        action_probabilities = np.exp(action_probabilities)

        action_probabilities  = action_probabilities/ float(np.sum(action_probabilities))

        return action_probabilities


    def dlnpi(self,features,theta,action):

        action_probabilities = self.softmax_policy(features,theta)

        result = np.zeros(shape=(self.features_length*self.num_actions,1), dtype=np.float32)

        for i in range(0,self.num_actions):

            start = self.features_length*i
            end = self.features_length*(i+1)

            #print start,end
            if i == action:
                result[start:end,] = (1 - action_probabilities[i][0]) * features
            else:
                result[start:end,] = -action_probabilities[i][0]* features

        #print result
        return result


    def find_fisher_info_matrix(self,features,theta,action):

        dlnpi = self.dlnpi(features,theta,action)

        dlnpi_t = dlnpi.T

        fim = dlnpi*dlnpi_t

        return fim

    def plot(self,standard_deviation_rewards):

        print standard_deviation_rewards.shape

        mean = np.mean(standard_deviation_rewards, axis=1)

        # print mean
        variance = np.mean(np.square(standard_deviation_rewards.T - mean).T, axis=1)

        # print variance
        std = np.sqrt(variance)
        # print std
        x = list(np.arange(0, self.num_episodes, 1))
        y = list(mean)
        err = list(std)

        print y[0]
        print variance

        #plt.axis((0, self.num_episodes, -1000, 0))
        plt.errorbar(x, y, yerr=err, fmt='-ro',ecolor='g')
        plt.xlabel('Episode')
        plt.ylabel('Expected return of reward')
        plt.savefig('gw_Expected_Return_INAC_%s_%s_%s_%s.png' % (str(self.num_trials),str(self.alpha_actor),str(self.alpha_critic),str(self.lamda)))
        plt.show()

def gw_sum_row(rewards_map,i,model):

        scipy.random.seed()
        grid_world = Gridworld(5)

        print model.features_length
        v = ev = np.zeros(shape=(model.features_length, 1), dtype=np.float32)
        '''
        e, theta , etheta , w === (length of phi(s).T,length of phi(s).T)T
        '''
        theta = etheta = w =  np.zeros(shape=(model.features_length*model.num_actions,1), dtype=np.float32)

        print("[Worker %d]is %f" % (os.getpid(), i))

        print 'running trial: ', i

        for j in range(0, model.num_episodes):
            print 'running episode: ', j
            rewards = 0

            # initialise the state as [-0.5,0]
            grid_world.initialise()
            features = model.get_features(grid_world)

            while (not grid_world.end_episode):

                action = model.take_action(features,theta)

                # On taking the action, new state is saved in Mountain Car environment.
                grid_world.Take_Action(action)
                r = grid_world.Get_Reward()
                rewards += model.gamma*r

                # If it reaches the terminal state, newFeatures = 0, and reinitialise etheta and ev
                if grid_world.end_episode:

                    print 'episode getting ended'
                    delta = r - np.dot(v.T,features)

                    ev = model.gamma * model.lamda *ev + features
                    v = v + model.alpha_critic * delta * ev
                    etheta = model.gamma *model.lamda *etheta + model.dlnpi(features,theta,action)

                    fim = model.find_fisher_info_matrix(features, theta, action)
                    w = w - np.dot(model.alpha_critic * fim, w) + model.alpha_critic * delta * etheta
                    theta += model.alpha_actor * w

                    ev = np.zeros(shape=(model.features_length, 1), dtype=np.float32)
                    etheta = np.zeros(shape=(model.features_length*model.num_actions,1), dtype=np.float32)

                else:
                    newFeatures = model.get_features(grid_world)
                    delta = r + model.gamma * np.dot(v.T,newFeatures) - np.dot(v.T,features)

                    ev = model.gamma * model.lamda * ev + features
                    v = v + model.alpha_critic * delta * ev

                    etheta = model.gamma * model.lamda * etheta + model.dlnpi(features, theta,action)

                    fim = model.find_fisher_info_matrix(features,theta,action)
                    w = w - np.dot(model.alpha_critic * fim, w) + model.alpha_critic * delta * etheta
                    theta += model.alpha_actor * w

                    features = newFeatures

            print 'rewards' , rewards
            rewards_map[j, i] = float(rewards)

        print 'expected reward and 0th weights of trial {0} : {1} , {2}'.format(i, rewards_map[0][i],rewards_map[-1][i],
                                                                                theta[0])