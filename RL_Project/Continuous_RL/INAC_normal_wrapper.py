import argparse
from Mc_INAC import *
from Pendulum_INAC import *
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--num_trials', '-n',required = True,help="Pass the num of episodes")
parser.add_argument('--lamda', '-l',required = True,help="lamda")
parser.add_argument('--gamma', '-g', help="pass gamma", required=True)
parser.add_argument('--alpha_actor', '-aa', help="pass alpha_actor", required=True)
parser.add_argument('--alpha_critic', '-ac', help="pass alpha_critic", required=True)
parser.add_argument('--fourier_order', '-fo', help="pass fourier order", required=True)
parser.add_argument('--domain', '-d', help="pass domain name : mc/pen", required=True)

args = parser.parse_args()

num_trials = int(args.num_trials)
lamda = args.lamda
gamma = args.gamma
alpha_actor = args.alpha_actor
alpha_critic = args.alpha_critic
fourier_order = args.fourier_order
domain_name = args.domain
num_episodes = 100

start = datetime.now()
count = []
rewards_map = np.zeros(dtype=np.float32,shape=(num_episodes,num_trials))

for i in range(0,int(num_trials)):

    if domain_name == 'mc':
        model = MC_INAC(num_episodes, int(num_trials), float(lamda), float(gamma), float(alpha_actor), float(alpha_critic),int(fourier_order))
        mc_sum_row(rewards_map,i,model)
    elif domain_name == 'pen':
        model = PEN_INAC(num_episodes, int(num_trials), float(lamda), float(gamma), float(alpha_actor), float(alpha_critic),
                     int(fourier_order))
        pen_sum_row(rewards_map, i, model, count)

end = datetime.now()
print 'total time taken is :', format(end - start)

with open('INAC_rewards.pickle', 'wb') as f:
    pickle.dump(rewards_map, f)


print rewards_map

if domain_name == 'mc':
    model.mc_plot(rewards_map)

elif domain_name == 'pen':
    model.pen_plot(rewards_map, count)
