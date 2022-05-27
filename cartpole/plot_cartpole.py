import numpy as np
import csv
from scipy.stats import sem
import matplotlib.pyplot as plt
from matplotlib import rc

fig, axs = plt.subplots(3)
x = np.linspace(0, 400, 400)
axs[0].grid()
axs[1].grid()
axs[2].grid()  

def smoothen(x, w):
    head = []
    for i in range(1,w):
        head.append(np.mean(x[0:i]))
    tail = np.convolve(x, np.ones(w), 'valid') / w
    return list(np.concatenate([head, list(tail)]))

def compute_correct_qmax_start(y, performance):
    sum = 0
    for i in range(abs(int(performance))):
        sum += 1*np.power(y,i)
    return sum

def plot(name, results, nr, color):
    axs[0].fill_between(x, results[1], results[2],label=name, alpha=.5, linewidth=0,color=color)
    axs[0].plot(x, results[0], linewidth=1, color=color)
    #axs[0].hline(y=200)
    axs[nr].fill_between(x, results[4], results[5],label=name, alpha=.5, linewidth=0, color=color)
    axs[nr].plot(x, results[3], linewidth=1, color=color)
    axs[nr].legend(loc='upper left')
    

def process(data, gamma):
    rewards = data[0]
    rewards = list(zip(*rewards))
    rewards_means = smoothen([np.mean(x) for x in rewards], 20)
    rewards_lower_std = smoothen([np.mean(x)-np.std(x) for x in rewards], 20)
    rewards_upper_std = smoothen([np.mean(x)+np.std(x) for x in rewards], 20)
    correctqmax = compute_correct_qmax_start(gamma, 200) 
    starting_states = data[1]
    starting_states = list(zip(*starting_states))
    startingstates_means = [np.mean(x) for x in starting_states]
    startingstates_stds = [np.std(x) for x in starting_states]
    startingstates_lower_std = [startingstates_means[idx]-x for idx,x in enumerate(startingstates_stds)]
    startingstates_upper_std = [startingstates_means[idx]+x for idx,x in enumerate(startingstates_stds)]
    processed_data = rewards_means[0:400], rewards_lower_std[0:400], rewards_upper_std[0:400], startingstates_means[0:400], startingstates_lower_std[0:400], startingstates_upper_std[0:400], correctqmax
    return processed_data

data = np.load('./data/cartpole_y999.npy',allow_pickle=True)
data = process(data, 0.999)
plot("$\gamma$ = .999", data,1,color='blue')

data = np.load('./data/cartpole_DDQN_y999.npy',allow_pickle=True)
data = process(data, 0.999)
plot("DDQN y= .999", data,1,color='green')

data = np.load('./data/cartpole_SCQL_y999.npy',allow_pickle=True)
data = process(data, 0.999)
plot("SCQL y= .999", data,1,color='yellow')

data = np.load('./data/cartpole_y97.npy',allow_pickle=True)
data = process(data, 0.97)
plot("$\gamma$ = .97", data,2,color='cyan')



axs[0].set_title("average rewards over time")
axs[1].set_title("degree of overestimation")
axs[2].set_title("degree of overestimation $\gamma$ =0.97")
axs[1].axhline(y=181.35,color = 'black', linestyle = '--')
axs[2].axhline(y=33.25,color = 'black', linestyle = '--')
plt.tight_layout()
plt.savefig('./graphs/cartpole_results.pdf')
plt.show()