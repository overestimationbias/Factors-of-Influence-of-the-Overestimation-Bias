import gym
import numpy as np
import random
import functions

def printQ(Q):
    for sum in range(4,22):
        for upcard in range(1,11):
            for ace in range(2):
                print("sum:",sum,"upcard:",upcard,"ace:",ace,"Stick:",Q[sum][upcard][ace][0], "Hit:", Q[sum][upcard][ace][1])


def run(num_episodes = 20000, y=.9, fixed_alpha = False, alpha=0.05, beta=4):
    #Load the environment
    env = gym.make('Blackjack-v1')
    # Set learning parameters
    #create lists to contain total rewards and steps per episode
    rList = []
    Qtable = []
    Q = np.zeros([32,11,2,2])
    Qprevious = np.zeros([32,11,2,2])
    Qbeta = np.zeros([32,11,2,2])
    times_visited = np.zeros([32,11,2])
    wins = 0
    ties = 0
    losses = 0
    for i in range(num_episodes):
        #print("new ep")
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
            times_visited[sum,dealer,ace] +=1
            a = np.argmax(Q[sum,dealer,ace,:])
            #epsilon = 1/np.sqrt(times_visited[sum,dealer,ace]+1)
            epsilon = 0.1
            if random.random()<epsilon:
                a = random.choice([0,1])
            #Get new state and reward from environment
            next_state,r,d,_ = env.step(a)
            if next_state[2] == False: next_ace=0
            if next_state[2] == True: next_ace=1
            #print("sum:",sum,"dealer:", dealer,"ace:", ace, "action:",a,"next_state:", next_state,"r:", r,"d:", d)
            #Update Q-Table with new knowledge
            if fixed_alpha == False: alpha = 1/times_visited[sum,dealer,ace]
            Qbeta[next_state[0],next_state[1],next_ace,a] = Q[next_state[0],next_state[1],next_ace,a] - beta*(Q[next_state[0],next_state[1],next_ace,a]-Qprevious[next_state[0],next_state[1],next_ace,a])
            ahat = np.argmax(Qbeta[next_state[0],next_state[1],next_ace,:])
            Qprevious[sum,dealer,ace,a] = Q[sum,dealer,ace,a]
            if d == False:
                Q[sum,dealer,ace,a] = Q[sum,dealer,ace,a] + alpha*(r + y*Q[next_state[0],next_state[1],next_ace,ahat] - Q[sum,dealer,ace,a])
            else: 
                Q[sum,dealer,ace,a] = Q[sum,dealer,ace,a] + alpha*(r-Q[sum,dealer,ace,a])
            Qtable.append(Q)
            rAll += r
            if r == 1: wins += 1
            elif r == 0: ties +=1
            elif r == -1: losses +=1
            s = next_state
            if d == True:
                break
        rList.append(rAll)
    return rList, wins, Qtable, functions.deviationFromBS(Q)