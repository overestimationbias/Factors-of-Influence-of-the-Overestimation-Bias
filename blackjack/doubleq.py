import gym
import numpy as np
import random
import functions

def run(num_episodes = 20000, y = .95):
    #Load the environment
    env = gym.make('Blackjack-v1')
    # Set learning parameters
    #create lists to contain total rewards and steps per episode
    #jList = []
    rList = []
    Qtable = []
    Q1 = np.zeros([32,11,2,2])
    Q2 = np.zeros([32,11,2,2])
    Qsum = np.zeros([32,11,2,2])
    times_visited = np.zeros([40,11,2,2])
    wins = 0
    ties = 0
    losses = 0
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
            epsilon = 1/np.sqrt(np.sum(times_visited[sum,dealer,ace])+1)
            #epsilon = 0.1
            a = np.argmax(Qsum[sum,dealer,ace,:])
            if np.random.rand()<epsilon:
                a = random.choice([0,1])
            #a = random.choice([0,1])
            #Get new state and reward from environment
            times_visited[sum,dealer,ace,a] +=1
            next_state,r,d,_ = env.step(a)
            next_sum = next_state[0]
            next_dealer=next_state[1]
            if next_state[2] == False: next_ace=0
            if next_state[2] == True: next_ace=1
            #print(sum, dealer, ace, next_state, next_ace)
            #Update Q-Table with new knowledge
            alpha = 1/times_visited[sum,dealer,ace,a]
            if d == True: 
                if np.random.rand() < 0.5:
                    Q1[sum,dealer,ace,a] = Q1[sum,dealer,ace,a] + alpha*(r- Q1[sum,dealer,ace,a])
                else:
                    Q2[sum,dealer,ace,a] = Q2[sum,dealer,ace,a] + alpha*(r- Q2[sum,dealer,ace,a])
            else:
                if np.random.rand() < 0.5:
                    a_star = np.argmax(Q1[sum,dealer,ace,:])
                    Q1[sum,dealer,ace,a] = Q1[sum,dealer,ace,a] + alpha*(r + y*Q2[next_sum,next_dealer,next_ace,a_star] - Q1[sum,dealer,ace,a])
                else:
                    a_star = np.argmax(Q2[sum,dealer,ace,:])
                    Q2[sum,dealer,ace,a] = Q2[sum,dealer,ace,a] + alpha*(r + y*Q1[next_sum,next_dealer,next_ace,a_star] - Q2[sum,dealer,ace,a])
            Qsum[sum,dealer,ace,a] = Q1[sum,dealer,ace,a] + Q2[sum,dealer,ace,a]
            Qtable.append(Qsum/2)
            rAll += r
            if r == 1: wins += 1
            elif r == 0: ties +=1
            elif r == -1: losses +=1
            s = next_state
            if d == True:
                break
        #jList.append(j)
        rList.append(rAll)
    #print(Q)
    #print ("Score over time: " +  str(sum(rList)/num_episodes))
    #print ("Final Q-Table Values")
    #print (Q)
    return rList, wins, Qtable, functions.deviationFromBS(Q1)