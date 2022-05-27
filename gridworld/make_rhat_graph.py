import numpy as np
import matplotlib.pyplot as plt

means = []
means_std = []
labels = []

colors=["blue", "green", "yellow", "purple", "purple", "purple", "purple", "purple", "purple", "purple", "purple", "purple", "purple", "purple", "purple", ]
def get_value(value):
    filename = './data/rhat' + str(value) + '.npy'
    data = np.load(filename, allow_pickle=True)
    means.append(data[3][-1]- 0.36)
    means_std.append(data[4][-1]-data[3][-1]- 0.36)
    
for i in np.arange(10, 110, 10):
    get_value(i)
    labels.append('$x$' + str(i))

means.insert(0, np.load('./data/self correcting.npy', allow_pickle=True)[3][-1]-0.36)
means_std.insert(0, np.load('./data/self correcting.npy', allow_pickle=True)[4][-1]-[3][-1]-0.36)
labels.insert(0, 'SCQL')

means.insert(0, np.load('./data/double q.npy', allow_pickle=True)[3][-1]-0.36)
means_std.insert(0, np.load('./data/double q.npy', allow_pickle=True)[4][-1]-[3][-1]-0.36)
labels.insert(0, 'DQL')

means.insert(0, np.load('./data/regular.npy', allow_pickle=True)[3][-1]-0.36)
means_std.insert(0, np.load('./data/regular.npy', allow_pickle=True)[4][-1]-[3][-1]-0.36)
labels.insert(0, 'QL')

plt.grid()
plt.ylabel(r'$\max_{a\in \mathcal{A}}Q(s_0,a)-Q^*(s_0,a) $')
plt.bar(labels, means, yerr=means_std,capsize=2, color=colors)
plt.axhline(y = 0, color = 'black', linestyle = '--')
plt.xticks(fontsize=8)

plt.savefig('./graphs/rhat_study.pdf')
plt.show()
