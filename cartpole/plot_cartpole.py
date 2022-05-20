import numpy as np
import csv
from scipy.stats import sem
import matplotlib.pyplot as plt
from matplotlib import rc

fig, axs = plt.subplots(2)
x = np.linspace(0, 500, 500)

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
    axs[nr].axhline(y=1,color = 'black', linestyle = '--')
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
    startingstates_means = [np.mean(x)/correctqmax for x in starting_states]
    startingstates_stds = [np.std(x)/correctqmax for x in starting_states]
    startingstates_lower_std = [startingstates_means[idx]-x for idx,x in enumerate(startingstates_stds)]
    startingstates_upper_std = [startingstates_means[idx]+x for idx,x in enumerate(startingstates_stds)]
    processed_data = rewards_means, rewards_lower_std, rewards_upper_std, startingstates_means, startingstates_lower_std, startingstates_upper_std, correctqmax
    return processed_data

data = np.load('cartpole_y999.npy',allow_pickle=True)
data = process(data, 0.999)
plot("y= .999", data,1,color='blue')

data = np.load('cartpole_DDQN_y999.npy',allow_pickle=True)
data = process(data, 0.999)
plot("DDQN y= .999", data,1,color='green')

data = np.load('cartpole_SCQL_y999.npy',allow_pickle=True)
data = process(data, 0.999)
plot("SCQL y= .999", data,1,color='yellow')


data = np.load('cartpole_y97.npy',allow_pickle=True)
data = process(data, 0.97)
plot("y= .97", data,1,color='red')







axs[0].set_title("average rewards over time")
axs[0].get_xaxis().set_visible(False)
#axs[0].axhline(y = 500, color = 'black', linestyle = '--')

axs[1].set_title("degree of overestimation")
#axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.show()