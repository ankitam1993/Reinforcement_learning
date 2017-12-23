from MountainCar import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
from joblib import Parallel, delayed
import sys
import os , scipy
import dill

class Qlearning(object):

    def __init__(self,episodes,num_trials,epsilon,gamma,alpha,fourier_order):

        self.num_episodes = episodes
        self.fourier_order = fourier_order
        self.num_states = 2
        self.num_actions = 3
        self.features_length =  int(math.pow(self.fourier_order + 1,self.num_states))
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.num_trials = num_trials
        self.NaN = False
        #self.standard_deviation_rewards = np.zeros(shape=(self.num_episodes,self.num_trials))

    def calculate_features(self,action,mc):

        features = np.zeros(shape=(self.features_length, 1), dtype=np.float32)

        phi_s_a = np.zeros(shape=(self.num_actions * self.features_length, 1), dtype=np.float32)

        count = 0

        #print self.mountain_car.normalised_state

        for i in range(0, self.fourier_order+1):

            for j in range(0, self.fourier_order+1):
                f = np.array([i, j]).dot(mc.normalised_state)

                #print i , j , f , math.cos(math.pi *f)
                features[count, 0] = math.cos(math.pi *f)
                count +=1

        phi_s_a[action*self.features_length: action*self.features_length + self.features_length,:] = features

        return phi_s_a

    def compute_q_sa(self,action,w,mc):

        phi_s_a = self.calculate_features(action,mc)

        q_s_a = np.dot(w.transpose(),phi_s_a)[0][0]

        #print q_s_a , phi_s_a.shape

        return phi_s_a,q_s_a

    def epsilon_greedy(self,q_sas):

        random_number = np.random.rand()

        if random_number < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
             m = max(q_sas)
             a = []

             for i,x in enumerate(q_sas):
                if x == m:
                    a.append(i)

             #print m, a , q_sas

             z = np.random.choice(a,1)[0]
             return z


    def plot(self,standard_deviation_rewards):

        if self.NaN:
            standard_deviation_rewards = self.plot2(standard_deviation_rewards)

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
        plt.savefig('Expected_Return_%s.jpg' % ('ql_q2'))
        plt.show()

    # use this when weights are reaching to NaN
    def plot2(self,standard_deviation_rewards):

        trials_episode = standard_deviation_rewards.transpose()
        min_episode = self.num_episodes

        for tr in trials_episode:
            for i, episode in enumerate(tr):
                if episode == 0.0:
                    if i < min_episode:
                        min_episode = i
                    break

        print min_episode

        data = standard_deviation_rewards[0:min_episode]

        return data

def sum_row(rewards_map,i,model):

        flag = 0
        scipy.random.seed()
        mountain_car = MountainCar([-0.5, 0], [-1.2, 0.5], [-0.07, 0.07])

        # initialise w as zero vector , taking 2 states and 3 actions
        w = np.zeros(shape=(model.num_actions * model.features_length, 1), dtype=np.float32)

        print("[Worker %d]is %f" % (os.getpid(), i))

        print 'running trial: ', i

        for j in range(0, model.num_episodes):
            #print 'running episode: ', j

            rewards = 0

            # initialise the state as [-0.5,0]
            mountain_car.initialise()

            q_sas = []
            phi_sas = []

            # calculate phi and q
            for action in range(0, model.num_actions):
                phi_s_a, q_sa = model.compute_q_sa(action,w,mountain_car)
                q_sas.append(q_sa)
                phi_sas.append(phi_s_a)

            while (not mountain_car.end_episode):

                action = model.epsilon_greedy(q_sas)

                # On taking the action, new state is saved in Mountain Car environment.
                mountain_car.Take_Action(action)

                r = mountain_car.Get_Reward()
                rewards += r

                q_sa_sprime = []
                phi_sa_prime = []

                # If it reaches the terminal state, q(s_terminal,.) = 0, otherwise chose maximum action and
                # update w, s, a
                if mountain_car.end_episode:

                    q_sa_sprime = [0] * model.num_actions

                else:
                    # chose the maximum action
                    for a in range(0, model.num_actions):
                        phi_s_a, q_sa = model.compute_q_sa(a,w,mountain_car)
                        q_sa_sprime.append(q_sa)
                        phi_sa_prime.append(phi_s_a)

                max_q_sa = max(q_sa_sprime)

                w += model.alpha * (r + model.gamma * max_q_sa - q_sas[action]) * phi_sas[action]

                if math.isnan(w[0][0]):
                    print 'Divergence occured at episode for trial: :', j , i
                    flag = 1
                    model.Nan = True
                    break

                q_sas = []
                phi_sas = []

                for a in range(0, model.num_actions):
                    phi_s_a, q_sa = model.compute_q_sa(a, w, mountain_car)
                    q_sas.append(q_sa)
                    phi_sas.append(phi_s_a)

            # print type(w[0][0])

            if flag == 0:

                #print 'rewards, 0th weights of episode {0} are {1} , {2} '.format(j,rewards, w[0][0])

                rewards_map[j, i] = float(rewards)

            else:
                break

        print 'expected reward and 0th weights of trial {0} : {1} , {2}'.format(i, rewards_map[0][i],rewards_map[-1][i],
                                                                                w[0])