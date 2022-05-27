import numpy as np 
import matplotlib.pyplot as mpl


fig, ax = mpl.subplots()
axins = ax.inset_axes([0.4, 0.1, 0.5, 0.5])
mpl.grid()
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
    mpl.plot(x, results, c=color,linestyle='solid', label=name)
    mpl.fill_between(x, lower_avg, upper_avg, alpha=.3, linewidth=0, color= color)

def zoomed_plot(data, color, name):
    results, _, reward_error, _ = data
    lower_avg = results - reward_error
    upper_avg = results + reward_error
    results = smoothen(results)
    upper_avg = smoothen(upper_avg)
    lower_avg = smoothen(lower_avg)
    
    mpl.grid()
    axins.plot(x, results, c=color,linestyle='solid', label=name)
    axins.fill_between(x, lower_avg, upper_avg, alpha=.3, linewidth=2, color= color)

plot(np.load("./data/standard.npy", allow_pickle=True), "blue", "standard")
plot(np.load("./data/low_y.npy", allow_pickle=True), "grey", "low_y")
plot(np.load("./data/lr_05.npy", allow_pickle=True), "orange", "lr 0.05")
plot(np.load("./data/double_q.npy", allow_pickle=True), "green", "double q")
plot(np.load("./data/avgr.npy", allow_pickle=True), "purple", "avg r")
plot(np.load("./data/SCQL.npy", allow_pickle=True), "yellow", "SCQL")

zoomed_plot(np.load('./data/standard.npy', allow_pickle=True), "blue", "QL")
zoomed_plot(np.load('./data/low_y.npy', allow_pickle=True), "cyan", r"$\gamma=0.6$")
zoomed_plot(np.load('./data/lr_05.npy', allow_pickle=True), "red", r"$\alpha= 0.05$")
zoomed_plot(np.load('./data/double_q.npy', allow_pickle=True), "green", "DQL")
zoomed_plot(np.load('./data/avgr.npy', allow_pickle=True), "purple", "$\hat{r}$")
zoomed_plot(np.load('./data/SCQL.npy', allow_pickle=True), "yellow", "SCQL")


ax.axhline(y = -0.0451, color = 'black', linestyle = '--', label='Basic Strategy')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
x1, x2, y1, y2 = 80000, 100000, -0.12, -0.08
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
ax.indicate_inset_zoom(axins, edgecolor="black")
mpl.xlabel('Games')
mpl.ylabel('Reward')
mpl.savefig('./graphs/blackjack_results.pdf')
mpl.show()