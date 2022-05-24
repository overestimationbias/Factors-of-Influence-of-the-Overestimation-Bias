import gym
import numpy as np
import random
import sys
import functions
np.set_printoptions(threshold=sys.maxsize)


def run(num_episodes = 20000, avg_range = 10, y = .9):
    #Load the environment
    env = gym.make('Blackjack-v1')
    # Set learning parameters
    #create lists to contain total rewards and steps per episode
    rSum = 0
    rList = []
    Qtable = []
    Q = np.zeros([32,11,2,2])
    avg_r = np.zeros([32,11,2,2])
    times_visited = np.zeros([32,11,2,2])
    wins = 0
    ties = 0
    losses = 0
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rEpisode = 0
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
            a = np.argmax(Q[sum,dealer,ace,:])
            epsilon = 1/np.sqrt(np.sum(times_visited[sum,dealer,ace])+1)
            #epsilon = 0.1
            if random.random()<epsilon:
                if a == 1: a=0
                else: a = 1
            times_visited[sum,dealer,ace,a] +=1
            #Get new state and reward from environment
            next_state,r,d,_ = env.step(a)
            avg_r[sum,dealer,ace,a] += (r-avg_r[sum,dealer,ace,a])/avg_range
            if next_state[2] == False: next_ace=0
            if next_state[2] == True: next_ace=1
            #print("sum:",sum,"dealer:", dealer,"ace:", ace, "action:",a,"next_state:", next_state,"r:", r,"d:", d)
            #Update Q-Table with new knowledge
            alpha = 1/times_visited[sum,dealer,ace,a]
            if d==True:
                Q[sum,dealer,ace,a] += alpha*(avg_r[sum,dealer,ace,a]-Q[sum,dealer,ace,a])
            else:
                Q[sum,dealer,ace,a] += alpha*(avg_r[sum,dealer,ace,a]+y*np.max(Q[next_state[0],next_state[1],next_ace,:]) - Q[sum,dealer,ace,a])
            #print(Q[sum,dealer,ace,a])
            Qtable.append(Q)
            rEpisode += r
            if r == 1: wins += 1
            elif r == 0: ties +=1
            elif r == -1: losses +=1
            s = next_state
            if d == True:
                break
        rSum += rEpisode
        rList.append(rSum/(i+1))
    #print(Q)
    #np.savetxt('data.csv', Q, delimiter=',')
    #print ("Score over time: " +  str(sum(rList)/num_episodes))
    #print ("Final Q-Table Values")
    #print(times_visited)
    #functions.deviationFromBS(Q)
    #functions.printQandTimesVisited(Q,times_visited)
    return rList, wins, Qtable, functions.deviationFromBS(Q)