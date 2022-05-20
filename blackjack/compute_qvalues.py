import gym 
import numpy as np
import random

#prints a q table
def printQ(Q):
    for sum in range(4,22):
        for upcard in range(1,11):
            for ace in range(2):
                if not ace and sum<12:
                    print("sum:",sum,"upcard:",upcard,"ace:",ace,"Stick:",Q[sum][upcard][ace][0], "Hit:", Q[sum][upcard][ace][1])

#takes as import the state in form of sum, dealers upcard and ace=True/False and outputs the optimal action 
def basic_strategy(sum, dealer, ace):
    #"Hard totals" (hands without an ace)
    if not ace:
        if sum < 12: a=1
        if sum == 12:
            if dealer in [4,5,6]: a=0
            else: a=1
        if sum in range(13,17):
            if dealer in range(2,7): a=0
            else: a=1
        if sum >= 17: a=0
    #"soft totals" (hands with an ace)
    elif ace:
        a=0
        if sum <= 17: a=1
        if sum == 18 and dealer > 8: a=1
    return a

#compute bs q values using SARSA
def compute_bs_values(y=.9):
    #number of episodes: the larger, the better
    num_episodes = 100000
    #Load the environment
    env = gym.make('Blackjack-v1')
    #values will be saved in this table
    BS = np.zeros([32,11,2,2])
    times_visited = np.zeros([32,11,2,2])
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        #first state is random within the state space
        env.state = (random.randint(4,21),random.randint(1,10),random.randint(0,1))
        d = False
        j = 0
        #The Q-Table learning algorithm
        while j < 99:
            j+=1
            #extract information from the state tuple
            sum = s[0]
            dealer=s[1]
            if s[2] == False: ace=0
            if s[2] == True: ace=1
            #first action is random
            if j == 1:
                a = random.randint(0,1)
            else: 
                a = basic_strategy(sum, dealer, ace)
            times_visited[sum,dealer,ace,a] +=1
            #get next state, reward, and terminal state: y/n
            next_state,r,d,_ = env.step(a)
            #extract information from next state
            next_sum = next_state[0]
            next_dealer = next_state[1]
            if next_state[2] == False: next_ace=0
            if next_state[2] == True: next_ace=1
            #this is at+1
            next_action = basic_strategy(next_sum, next_dealer, next_ace)
            alpha = 1/times_visited[sum,dealer,ace,a]
            if d == False:
                BS[sum,dealer,ace,a] = BS[sum,dealer,ace,a] + alpha*(r+y*BS[next_sum,next_dealer,next_ace,next_action] - BS[sum,dealer,ace,a])
            else: BS[sum,dealer,ace,a] = BS[sum,dealer,ace,a] + alpha*(r - BS[sum,dealer,ace,a])
            s = next_state
            if d == True:
                break
    #np.save('BS_values', BS)
    printQ(BS)
    return BS

#MAIN
compute_bs_values()