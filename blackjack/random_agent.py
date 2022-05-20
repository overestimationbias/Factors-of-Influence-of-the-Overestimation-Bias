import gym
import numpy as np
import random

def run():
    #Load the environment
    env = gym.make('Blackjack-v1')
    num_episodes = 5000
    #create lists to contain total rewards and steps per episode
    rList = []
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
            a = np.random.choice([0,1])
            next_state,r,d,_ = env.step(a)
            rAll += r
            s = next_state
            if d == True:
                break
        rList.append(rAll)
    return np.sum(rList)/num_episodes