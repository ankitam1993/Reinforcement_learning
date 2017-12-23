import argparse
from Mc_INAC import *
from GW_inac import *
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--num_trials', '-n',required = True,help="Pass the num of episodes")
parser.add_argument('--lamda', '-l',required = True,help="lamda")
parser.add_argument('--gamma', '-g', help="pass gamma", required=True)
parser.add_argument('--alpha_actor', '-aa', help="pass alpha_actor", required=True)
parser.add_argument('--alpha_critic', '-ac', help="pass alpha_critic", required=True)
parser.add_argument('--fourier_order', '-fo', help="pass fourier order", required=True)
parser.add_argument('--domain', '-d', help="pass domain name : mc/gw", required=True)

args = parser.parse_args()

num_trials = int(args.num_trials)
lamda = args.lamda
gamma = args.gamma
alpha_actor = args.alpha_actor
alpha_critic = args.alpha_critic
domain_name = args.domain
fourier_order = args.fourier_order
num_episodes = 100

start = datetime.now()

folder = tempfile.mkdtemp()

sums_name = os.path.join(folder, 'rewards')

# Pre-allocate a writeable shared memory map as a container for the
# results of the parallel computation
rewards_map = np.memmap(sums_name, dtype=np.float32,
                         shape=(num_episodes,num_trials), mode='w+')

# Fork the worker processes to perform computation concurrently
if domain_name == 'mc':
    model = MC_INAC(num_episodes,int(num_trials),float(lamda),float(gamma),float(alpha_actor),float(alpha_critic),int(fourier_order))
    Parallel(n_jobs=4)(delayed(mc_sum_row)(rewards_map, i, model)
                       for i in range(0, num_trials))
elif domain_name == 'gw':
    model = GW_INAC(num_episodes, int(num_trials), float(lamda), float(gamma), float(alpha_actor), float(alpha_critic),
                    int(fourier_order))
    Parallel(n_jobs=4)(delayed(gw_sum_row)(rewards_map,i,model)
                           for i in range(0,num_trials))



end = datetime.now()
print 'total time taken is :', format(end - start)

with open('INAC_rewards.pickle', 'wb') as f:
    pickle.dump(rewards_map, f)


print rewards_map

model.plot(rewards_map)
