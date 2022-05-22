import regularq
import avgr
import doubleq
import doubleq_avg
import basic_strategy
import random_agent
import self_correcting_ql
import functions
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import sem

no_of_experiments = 50
num_episodes = 20000
smoothing = 3000


def perform(function):
    return function()

def run(name, function):
    results = []
    final_average = []
    win_percentage = []
    deviations = []
    overestimations = []
    overestimation = []
    for i in range (no_of_experiments):
        print(i)
        data, wins, Q, deviation = perform(function)
        data = list(functions.moving_average(data,smoothing))
        data = np.round(data, 4)
        Q = np.round(Q, 4)
        results.append(data)
        win_percentage.append(wins)
        final_average.append(data[-1])  
        deviations.append(deviation)
        overestimation = [functions.compute_relative_overestimation(data[i], Q[i]) for i in range(num_episodes-smoothing)]
        overestimations.append(overestimation)
        #functions.make_table(Q)
        #functions.printStrategy(Q)
        #functions.printQ(Q)
        #functions.deviationFromBS(Q)
        #print(f"{name} {i}: wins:  {wins}, deviations: {deviation}, overestimation:{overestimation}")
    error = sem(final_average)
    results = np.average(results, axis=0)
    overestimations = list(zip(*overestimations))
    overestimations_avg = [np.mean(x) for x in overestimations]
    print(f"{name}: average win rate: {np.mean(win_percentage)/num_episodes}, average deviations: {np.mean(deviations)}")
    return (results, overestimations_avg, error)
    

np.save("standard_eps01", run(name="standard", function= lambda: regularq.run(num_episodes,y=1)))
np.save("low_y_eps01", run(name="low y", function= lambda: regularq.run(num_episodes,y=0.6)))
np.save("lr_05_eps01", run(name="lr 0.05", function= lambda: regularq.run(num_episodes,y=1, fixed_alpha=True, alpha=0.05)))
np.save("double_q_eps01", run(name="double q", function= lambda: doubleq.run(num_episodes,y=1)))
np.save("avgr_eps01", run(name="avgr", function= lambda: avgr.run(num_episodes,y=1)))
np.save("SCQL_eps01", run(name="SCQL", function= lambda: self_correcting_ql.run(num_episodes,y=1)))

