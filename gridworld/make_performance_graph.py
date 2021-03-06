import numpy as np
import matplotlib.pyplot as plt


plt.grid()

x = np.linspace(0, 10000, 10000)
font = {'size'   : 12}
plt.rc('font', **font)
    
def plot_performance(name, results, color):
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.fill_between(x, results[1], results[2], alpha=.2, linewidth=0, color= color)
    plt.plot(x, results[0], label=name, linewidth=2.5, color= color)
    
def plot(name, results, colour):
    plot_performance(name, results, colour)
    plt.ylim([-1.5, 0.3])
    plt.axhline(y = 0.2, color = 'black', linestyle = '--')
    plt.legend(loc='lower right', fancybox=True)

regular = np.load('./data/regular.npy',allow_pickle=True)
plot("QL", regular, "blue")
doubleq = np.load('./data/double q.npy',allow_pickle=True)
plot("DQL", doubleq, "green")
self_correcting = np.load('./data/self correcting.npy',allow_pickle=True)
plot("SCQL", self_correcting, "yellow")
running_avg = np.load('./data/rhat70.npy',allow_pickle=True)
plot("$\hat{r}$", running_avg, "purple")
fixed_alpha = np.load('./data/a0.05.npy',allow_pickle=True)
plot("static a", fixed_alpha, "grey")
regular = np.load('./data/y0.6.npy',allow_pickle=True)
plot("low $\gamma$", regular, "red")


plt.savefig("./graphs/OE_graph_2.pdf")
plt.show()
