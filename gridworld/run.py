from gridworld import iteration
from good_world_bad_world import gwbd_iteration
import numpy as np
import matplotlib.pyplot as plt

UP = 0
RIGHT = 1
DOWN = 2 
LEFT = 3 

DETERMINISTIC = 0
HIGH_VARIANCE_GAUSSIAN = 1 
LOW_VARIANCE_GAUSSIAN = 2
BERNOULLI = 3
REVERSED_BERNOULLI = 4

STANDARD = 0
DOUBLE_Q = 1
STANDARD_AND_REWARDTABLE = 2
STANDARD_AND_RUNNINGAVG = 3
DOUBLE_Q_RUNNINGAVG = 4
SARSA = 5
EXPECTED_SARSA = 6
SELF_CORRECTING = 7 


def process(results):
    rewards_means = [np.mean(x) for x in results[0]]
    rewards_lower_std = [np.mean(x)-np.std(x) for x in results[0]]
    rewards_upper_std = [np.mean(x)+np.std(x) for x in results[0]]
    startingstates_means = [np.mean(x) for x in results[1]]
    startingstates_lower_std = [np.mean(x)-np.std(x) for x in results[1]]
    startingstates_upper_std = [np.mean(x)+np.std(x) for x in results[1]]
    return rewards_means, rewards_lower_std, rewards_upper_std, startingstates_means, startingstates_lower_std, startingstates_upper_std, results[2]

""""
results = iteration(STANDARD, BERNOULLI, discount_factor=0.95, dynamic_y=False,running_avg_range=1,dynamic_alpha=True,alpha=0)
results = process(results)
np.save("regular",results)

results = iteration(DOUBLE_Q, BERNOULLI, discount_factor=0.95, dynamic_y=False,running_avg_range=1,dynamic_alpha=True,alpha=0)
results = process(results)
np.save("double q",results)

results = iteration(STANDARD_AND_RUNNINGAVG, BERNOULLI, discount_factor=0.95, dynamic_y=False,running_avg_range=100,dynamic_alpha=True,alpha=0)
results = process(results)
np.save("running average",results)

results = iteration(STANDARD, BERNOULLI, discount_factor=0.6, dynamic_y=False,running_avg_range=1,dynamic_alpha=True,alpha=0.01)
results = process(results)
np.save("fregular low y",results)

results = iteration(STANDARD, BERNOULLI, discount_factor=0.95, dynamic_y=False,running_avg_range=1,dynamic_alpha=False,alpha=0.01)
results = process(results)
np.save("Fixed alpha 0.01",results)

results = iteration(STANDARD_AND_RUNNINGAVG, BERNOULLI, discount_factor=0.6, dynamic_y=False,running_avg_range=100,dynamic_alpha=False,alpha=0.01)
results = process(results)
np.save("combination",results)
"""

results = iteration(STANDARD, BERNOULLI, discount_factor=0.95, dynamic_y=False,running_avg_range=100,dynamic_alpha=False,alpha=0.05)
results = process(results)
np.save("a0.05",results)

results = iteration(STANDARD, BERNOULLI, discount_factor=0.95, dynamic_y=False,running_avg_range=100,dynamic_alpha=False,alpha=0.01)
results = process(results)
np.save("a0.01",results)



