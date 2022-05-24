import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import sem

plt.grid()
x = np.linspace(0, 100000, 100000)

def smoothen(x, w=300):
    head = []
    for i in range(1,w):
        head.append(np.mean(x[0:i]))
    tail = np.convolve(x, np.ones(w), 'valid') / w
    return list(np.concatenate([head, list(tail)]))

def plot(data, color, name):
    results, _, reward_error, _ = data
    lower_avg = results - reward_error
    upper_avg = results + reward_error
    results = smoothen(results)
    upper_avg = smoothen(upper_avg)
    lower_avg = smoothen(lower_avg)
    plt.plot(x, results, c=color,linestyle='solid', label=name)
    plt.fill_between(x, lower_avg, upper_avg, alpha=.3, linewidth=0, color= color)

plot(np.load("./data/standard.npy", allow_pickle=True), "blue", "standard")
plot(np.load("./data/low_y.npy", allow_pickle=True), "grey", "low_y")
plot(np.load("./data/lr_05.npy", allow_pickle=True), "orange", "lr 0.05")
plot(np.load("./data/double_q.npy", allow_pickle=True), "green", "double q")
plot(np.load("./data/avgr.npy", allow_pickle=True), "purple", "avg r")
plot(np.load("./data/SCQL.npy", allow_pickle=True), "yellow", "SCQL")


plt.title(r'Performance of difference algorithms')
plt.axhline(y = -0.0451, color = 'black', linestyle = '--')
plt.text(s="optimum",y=-0.053, x=0)
plt.legend(loc='lower right', ncol=3)
plt.show()