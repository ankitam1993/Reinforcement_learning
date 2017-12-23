import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":

    filename = sys.argv[1]
    data = np.loadtxt(filename, delimiter=",", skiprows=1)

    plt.figure()
    plt.errorbar(np.asarray(range(len(data))), data[:,0], yerr=data[:,1], marker='^', ecolor='r')
    plt.show()