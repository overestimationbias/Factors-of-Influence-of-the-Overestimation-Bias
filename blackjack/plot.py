import functions
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import sem


def plot(data, color, name):
    results, overestimations, error = data
    plt.plot(results, c=color,linestyle='solid', label=name)
    plt.errorbar((len(results)-1),results[-1],error, linewidth=2, capsize=6, c=color)
    


plot(np.load("./data/standard.npy", allow_pickle=True), "blue", "standard")
plot(np.load("./data/low_y.npy", allow_pickle=True), "grey", "low_y")
plot(np.load("./data/lr_05.npy", allow_pickle=True), "orange", "lr 0.05")
plot(np.load("./data/double_q.npy", allow_pickle=True), "green", "double q")
plot(np.load("./data/avgr.npy", allow_pickle=True), "purple", "avg r")
plot(np.load("./data/SCQL.npy", allow_pickle=True), "yellow", "SCQL")

"""
plot(np.load("standard_eps01.npy", allow_pickle=True), "blue", "standard")
plot(np.load("low_y_eps01.npy", allow_pickle=True), "grey", "low_y")
plot(np.load("lr_05_eps01.npy", allow_pickle=True), "orange", "lr 0.05")
plot(np.load("double_q_eps01.npy", allow_pickle=True), "green", "double q")
plot(np.load("avgr_eps01.npy", allow_pickle=True), "purple", "avg r")
plot(np.load("SCQL_eps01.npy", allow_pickle=True), "yellow", "SCQL")
"""

plt.title(r'Performance of difference algorithms')
plt.axhline(y = -0.08, color = 'black', linestyle = '--')
plt.text(s="optimum",y=-0.09, x=0)
plt.legend(loc='lower right', ncol=3)
plt.show()