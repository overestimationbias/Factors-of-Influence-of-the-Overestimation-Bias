import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10000, 10000)
#font = {'size'   : 8}
#plt.rc('font', **font)


def plot_overestimation(name, results, color):
    plt.fill_between(x, results[4], results[5],label=name, alpha=.3, linewidth=0, color= color)
    plt.plot(x, results[3], linewidth=1.5, color = color)


    
def plot(name, results, colour):
        plot_overestimation(name, results, colour)
        plt.title(r'average $\max_{a}$ $Q(s_0,a)$ for $\gamma=0.95$')
        plt.axhline(y = 0.36, color = 'black', linestyle = '--')
        plt.legend(loc='lower center', fancybox=True, ncol=7)


regular = np.load('./data/regular.npy',allow_pickle=True)
plot("QL", regular, "blue")
doubleq = np.load('./data/double q.npy',allow_pickle=True)
plot("DQL", doubleq, "green")
self_correcting = np.load('./data/self correcting.npy',allow_pickle=True)
plot("SCQL", self_correcting, "yellow")
running_avg = np.load('./data/running average.npy',allow_pickle=True)
plot("$\hat{r}$", running_avg, "purple")
fixed_alpha = np.load('./data/fixed alpha.npy',allow_pickle=True)
plot("static a", fixed_alpha, "green")


plt.savefig("OE_graph")
plt.show()
