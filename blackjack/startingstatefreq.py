import gym 
import numpy as np
import random

def printQ(Q):
    for sum in range(4,22):
        for upcard in range(1,11):
            for ace in range(2):
                print("sum:",sum,"upcard:",upcard,"ace:",ace,"Freq:",Q[sum][upcard][ace])

#this function computes the frequency of starting states. 
env = gym.make('Blackjack-v1')
freq = np.zeros([32,11,2])
for i in range(10000000):
        s = env.reset()
        sum = s[0]
        dealer=s[1]
        if s[2] == False: ace=0
        if s[2] == True: ace=1
        freq[sum,dealer,ace] += 1

freq /= np.sum(freq)
printQ(freq)
np.save("SSfreq",freq)
