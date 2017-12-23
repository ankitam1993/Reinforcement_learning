from pendulum import *
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
from joblib import Parallel, delayed
import sys
import os , scipy , math

class PEN_INAC(object):

    def __init__(self,episodes,num_trials,lamda,gamma,alpha_actor,alpha_critic,fourier_order):

        self.num_trials = num_trials
        self.num_episodes = episodes
        self.lamda = lamda
        self.gamma = gamma
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic

        self.fourier_order = fourier_order
        self.num_states = 2 # Position and velocity
        self.features_length =  int(math.pow(self.fourier_order + 1,self.num_states))

    def make_features(self,pen):

        features = np.zeros(shape=(self.features_length, 1), dtype=np.float64)

        count = 0

        #print pen.normalised_state

        for i in range(0, self.fourier_order+1):
            for j in range(0, self.fourier_order+1):

                f = np.array([i, j]).dot(pen.normalised_state)

                #print i , j , f , math.cos(math.pi *f)
                features[count, 0] = math.cos(math.pi *f)
                count +=1

        return features

    # Take action according to normal/gaussian distribution
    def gaussian_distribution(self,mu,sigma):

        action = np.random.normal(mu, sigma)

        #print mu,sigma
        return action

    def create_theta(self):

        theta_mu = np.zeros(shape=(self.features_length, 1), dtype=np.float64)
        theta_sigma = np.zeros(shape=(self.features_length, 1), dtype=np.float64)

        theta_mu_t = theta_mu.T
        theta_sigma_t = theta_sigma.T

        theta_t = np.ndarray(shape=(1,2*self.features_length))
        theta_t[0,0:self.features_length] = theta_mu_t
        theta_t[0,self.features_length:] = theta_sigma_t
        theta = theta_t.T

        # shape of theta is : 2*features_length x 1
        return theta

    def calculate_mu_sigma(self,theta,features):

        theta_mu = theta[0:self.features_length,]
        theta_sigma = theta[self.features_length:,]

        mu = np.dot(theta_mu.T,features)[0][0]
        sigma = np.e**(np.dot(theta_sigma.T,features)[0][0])

        return mu,sigma

    def dlnpi(self,features, mu,sigma,action):

        num = (action - mu)*features

        #print 'sigma' , sigma
        var = sigma**2 #+ 0.005

        step = var*self.alpha_actor

        dlnpi_mu = (step * num) / float(var)

        dlnpi_sigma = step * (float((action - mu) ** 2) / var - 1) * features

        #print var
        #dlnpi_mu = num/float(var)

        #dlnpi_sigma = (float((action-mu)**2)/var - 1)*features

        dlnpi_t = np.ndarray(shape=(1, 2 * self.features_length))
        dlnpi_t[0, 0:self.features_length] = dlnpi_mu.T
        dlnpi_t[0, self.features_length:] = dlnpi_sigma.T
        dlnpi = dlnpi_t.T

        #print 'dlnpi' , dlnpi
        # shape of dlnpi is : 2*features_length x 1
        return dlnpi


    def find_fisher_info_matrix(self,features, mu, sigma,action):

        dlnpi = self.dlnpi(features, mu, sigma, action)

        dlnpi_t = dlnpi.T

        fim = dlnpi*dlnpi_t

        return fim

    def pen_plot(self,standard_deviation_rewards,count):

        print standard_deviation_rewards.shape

        rewards_map = np.zeros(dtype=np.float32, shape=(self.num_episodes, self.num_trials-len(count)))

        print rewards_map.shape

        for i in range(0,self.num_episodes):
            tr_count = 0
            for j in range(0,self.num_trials):
                if j in count:
                    print tr_count, count , j
                    assert standard_deviation_rewards[i][j] == False
                else:
                    print tr_count, j
                    rewards_map[i][tr_count] = standard_deviation_rewards[i][j]
                    tr_count += 1

        mean = np.mean(rewards_map, axis=1)

        # print mean
        variance = np.mean(np.square(rewards_map.T - mean).T, axis=1)

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
        plt.savefig('Expected_Return_INAC_%s_%s_%s_%s.png' % (str(self.num_trials),str(self.alpha_actor),str(self.alpha_critic),str(self.lamda)))
        plt.show()

def pen_sum_row(rewards_map,i,model,count):

        scipy.random.seed()
        pendulum = Pendulum([-1*math.pi, math.pi], [-3*math.pi, 3*math.pi])

        v = ev = np.zeros(shape=(model.features_length, 1), dtype=np.float64)
        '''
        e, theta , etheta , w === (length of phi(s).T,length of phi(s).T)T
        '''
        theta = etheta = w = model.create_theta()

        print("[Worker %d]is %f" % (os.getpid(), i))

        print 'running trial: ', i

        try:
            for j in range(0, model.num_episodes):
                print 'running episode: ', j
                rewards = 0

                pendulum.initialise()
                features = model.make_features(pendulum)

                while (not pendulum.end_episode):

                    mu,sigma = model.calculate_mu_sigma(theta,features)

                    #print 'mu , sigma' , mu , sigma
                    action = model.gaussian_distribution(mu,sigma)

                    #print 'action' , action

                    if sigma == 0.0:

                        print 'sigma reached 0'
                        print 'features' , features
                        print 'theta' , theta
                        raise Exception('general exceptions not caught by specific handling')

                    # On taking the action, new state is saved in Mountain Car environment.
                    pendulum.Take_Action(action)
                    r = pendulum.Get_Reward()
                    rewards += model.gamma*r

                    # If it reaches the terminal state, newFeatures = 0, and reinitialise etheta and ev
                    if pendulum.end_episode:

                        print 'episode getting ended'
                        delta = r - np.dot(v.T,features)

                        ev = model.gamma * model.lamda *ev + features
                        v = v + model.alpha_critic * delta * ev
                        etheta = model.gamma *model.lamda *etheta + model.dlnpi(features, mu,sigma,action)

                        fim = model.find_fisher_info_matrix(features, mu, sigma,action)
                        w = w - np.dot(model.alpha_critic * fim, w) + model.alpha_critic * delta * etheta
                        theta += model.alpha_actor * w

                        ev = np.zeros(shape=(model.features_length, 1), dtype=np.float64)
                        etheta = model.create_theta()

                    else:
                        newFeatures = model.make_features(pendulum)
                        delta = r + model.gamma * np.dot(v.T,newFeatures) - np.dot(v.T,features)

                        ev = model.gamma * model.lamda * ev + features
                        v = v + model.alpha_critic * delta * ev
                        etheta = model.gamma * model.lamda * etheta + model.dlnpi(features, mu, sigma,action)

                        fim = model.find_fisher_info_matrix(features, mu, sigma,action)
                        w = w - np.dot(model.alpha_critic * fim, w) + model.alpha_critic * delta * etheta
                        theta += model.alpha_actor * w

                        features = newFeatures

                print 'rewards' , rewards
                rewards_map[j, i] = float(rewards)
        except (ValueError,Exception):

            print 'ignoring trial number: ' , i
            count.append(i)
            for k in range(0,model.num_episodes):
                rewards_map[k, i] = False


        print 'expected reward and 0th weights of trial {0} : {1} , {2}'.format(i, rewards_map[0][i],rewards_map[-1][i],
                                                                                w[0])