import numpy as np
from scipy.stats import sem


arr = [[1,2,3],[-3,2,3],[5,2,3]]

sem = sem(arr, axis = 0)
print(sem)