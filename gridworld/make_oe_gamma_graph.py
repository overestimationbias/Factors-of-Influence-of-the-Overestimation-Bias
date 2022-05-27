from matplotlib import pyplot as plt 
import numpy as np 

x = np.linspace(0, 10000, 10000)
font = {'size'   : 12}
plt.rc('font', **font)

def plot_overestimation(name, results, color):
    plt.fill_between(x, results[4], results[5], alpha=.3, linewidth=0, color= color)
    plt.plot(x, results[3], linewidth=2.5, label=name, color = color)

def plot(name, results, colour):
        plot_overestimation(name, results, colour)
        plt.grid()
        plt.xlabel('Training Steps')
        plt.ylabel(r'$\max_{a\in\mathcal{A}}$ $Q(s_0,a)$')
        plt.axhline(y = 0.248, color = 'black', linestyle = '--')
        plt.legend(loc='upper right', fancybox=True)


regular = np.load('./data/y0.6.npy',allow_pickle=True)
plot(r'$\gamma=0.6$', regular, "cyan")

plt.savefig("./graphs/OE_graph_gamma.pdf")
plt.show()