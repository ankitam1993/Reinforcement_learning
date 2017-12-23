import argparse
from Mc_Qlearning_parallel import *
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--num_trials', '-n',required = True,help="Pass the num of episodes")
parser.add_argument('--epsilon', '-e',required = True,help="epsilon")
parser.add_argument('--gamma', '-g', help="pass gamma", required=True)
parser.add_argument('--alpha', '-a', help="pass alpha", required=True)
parser.add_argument('--fourier_order', '-fo', help="pass fourier order", required=True)

args = parser.parse_args()

num_trials = int(args.num_trials)
epsilon = args.epsilon
gamma = args.gamma
alpha = args.alpha
fourier_order = args.fourier_order
num_episodes = 200

model = Qlearning(num_episodes,int(num_trials),float(epsilon),float(gamma),float(alpha),int(fourier_order))


start = datetime.now()

folder = tempfile.mkdtemp()

sums_name = os.path.join(folder, 'rewards')

# Pre-allocate a writeable shared memory map as a container for the
# results of the parallel computation
rewards_map = np.memmap(sums_name, dtype=np.float32,
                         shape=(num_episodes,num_trials), mode='w+')

# Fork the worker processes to perform computation concurrently
Parallel(n_jobs=4)(delayed(sum_row)(rewards_map,i,model)
                           for i in range(0,num_trials))


end = datetime.now()
print 'total time taken is :', format(end - start)

with open('ql_rewards.pickle', 'wb') as f:
    pickle.dump(rewards_map, f)


print rewards_map

model.plot(rewards_map)
