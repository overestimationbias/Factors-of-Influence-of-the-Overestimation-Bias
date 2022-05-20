import numpy as np
import matplotlib.pyplot as plt

means = []
means_std = []
labels = []
colors=["yellow", "green", "red","blue","red","red","red","red","red","red","red","red","red","red","red","red","red","red","red","red","red","red","red","red",]

def get_value(value):
    filename = './data/y' + str(value) + '.npy'
    data = np.load(filename, allow_pickle=True)
    q_star = -1 - value - pow(value,2) - pow(value,3) + 5*pow(value,4)
    means.append(data[3][-1]-q_star)
    means_std.append(data[4][-1]-data[3][-1]-q_star)

   

for i in np.arange(1, 0, -0.05):
    i = round(i,2) 
    get_value(i)
    labels.append('$\gamma$' + str(i))

means.insert(0, np.load('./data/double q.npy', allow_pickle=True)[3][-1]-0.36)
means_std.insert(0, np.load('./data/double q.npy', allow_pickle=True)[4][-1]-[3][-1]-0.36)
labels.insert(0, 'DQL')
means.insert(0, np.load('./data/self correcting.npy', allow_pickle=True)[3][-1]-0.36)
means_std.insert(0, np.load('./data/self correcting.npy', allow_pickle=True)[4][-1]-[3][-1]-0.36)
labels.insert(0, 'SCQL')
plt.bar(labels, means, yerr=means_std,capsize=2, color= colors)
plt.axhline(y = 0, color = 'black', linestyle = '--')
plt.xticks(fontsize=8)
plt.show()
