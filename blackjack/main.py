import regularq
import avgr
import doubleq
import doubleq_avg
import random_agent
import self_correcting_ql
import functions
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import sem

no_of_experiments = 50
num_episodes = 100000

def perform(function):
    return function()

def run(name, function):
    results = []
    overestimations = []
    for i in range (no_of_experiments):
        print(i)
        data, _, Q, _ = perform(function)
        data = np.round(data, 4)
        Q = np.round(Q, 4)
        results.append(data)
        overestimation = [functions.compute_relative_overestimation(data[i], Q[i]) for i in range(num_episodes)]
        overestimations.append(overestimation)
    results_error = sem(results, axis = 0)
    results = np.average(results, axis=0)
    oe_error = sem(overestimations, axis = 0)
    overestimations_avg = np.average(overestimations, axis=0)
    print(f"{name}: done")
    return (results, overestimations_avg, results_error, oe_error)

np.save("standard_err_goodaverage", run(name="standard", function= lambda: regularq.run(num_episodes,y=1)))
np.save("low_y_err_goodaverage", run(name="low y", function= lambda: regularq.run(num_episodes,y=0.6)))
np.save("lr_05_err_goodaverage", run(name="lr 0.05", function= lambda: regularq.run(num_episodes,y=1, fixed_alpha=True, alpha=0.05)))
np.save("double_q_err_goodaverage", run(name="double q", function= lambda: doubleq.run(num_episodes,y=1)))
np.save("avgr_err_goodaverage", run(name="avgr", function= lambda: avgr.run(num_episodes,y=1)))
np.save("SCQL_err_goodaverage", run(name="SCQL", function= lambda: self_correcting_ql.run(num_episodes,y=1, beta=2)))