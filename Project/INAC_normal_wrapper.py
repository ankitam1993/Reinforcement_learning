import argparse
from Mc_INAC import *
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--num_trials', '-n',required = True,help="Pass the num of episodes")
parser.add_argument('--lamda', '-l',required = True,help="lamda")
parser.add_argument('--gamma', '-g', help="pass gamma", required=True)
parser.add_argument('--alpha_actor', '-aa', help="pass alpha_actor", required=True)
parser.add_argument('--alpha_critic', '-ac', help="pass alpha_critic", required=True)
parser.add_argument('--fourier_order', '-fo', help="pass fourier order", required=True)

args = parser.parse_args()

num_trials = int(args.num_trials)
lamda = args.lamda
gamma = args.gamma
alpha_actor = args.alpha_actor
alpha_critic = args.alpha_critic
fourier_order = args.fourier_order
num_episodes = 100

model = INAC(num_episodes,int(num_trials),float(lamda),float(gamma),float(alpha_actor),float(alpha_critic),int(fourier_order))

start = datetime.now()

rewards_map = np.zeros(dtype=np.float32,
                         shape=(num_episodes,num_trials))

sum_row(rewards_map,0,model)

end = datetime.now()
print 'total time taken is :', format(end - start)

with open('INAC_rewards.pickle', 'wb') as f:
    pickle.dump(rewards_map, f)


print rewards_map

model.plot(rewards_map)
