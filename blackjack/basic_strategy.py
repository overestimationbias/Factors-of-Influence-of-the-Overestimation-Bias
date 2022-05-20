import gym
import numpy as np
import random

def run(num_episodes = 20000):
    #Load the environment
    env = gym.make('Blackjack-v1')
    #create lists to contain total rewards and steps per episode
    rList = []
    wins = 0
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Table learning algorithm
        while j < 99:
            j+=1
            #Choose an action by greedily (with noise) picking from Q table
            sum = s[0]
            dealer=s[1]
            if s[2] == False: ace=0
            if s[2] == True: ace=1
            #Hard totals
            if not ace:
                if sum < 12: a=1
                if sum == 12:
                    if dealer in [4,5,6]: a=0
                    else: a=1
                if sum in range(13,17):
                    if dealer in range(2,7): a=0
                    else: a=1
                if sum >= 17: a=0
            #Soft totals
            elif ace:
                a=0
                if sum <= 17: a=1
                if sum == 18 and dealer > 8: a=1
            next_state,r,d,_ = env.step(a)
            rAll += r
            s = next_state
            if d == True:
                break
        rList.append(rAll)
        if r == 1: wins+=1
    return rList, wins