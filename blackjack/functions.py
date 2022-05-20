from re import A
import numpy as np
import gym
import random
import csv

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def printQ(Q):
    for sum in range(4,22):
        for upcard in range(1,11):
            for ace in range(2):
                print("sum:",sum,"upcard:",upcard,"ace:",ace,"Stick:",Q[sum][upcard][ace][0], "Hit:", Q[sum][upcard][ace][1])

def printQandTimesVisited(Q, timesvisited):
    for sum in range(4,22):
        for upcard in range(1,11):
            for ace in range(2):
                print("sum:",sum,"upcard:",upcard,"ace:",ace,"Stick:",Q[sum][upcard][ace][0],"visited:",timesvisited[sum][upcard][ace][0], "Hit:", Q[sum][upcard][ace][1],"visited:", timesvisited[sum][upcard][ace][1])


def printStrategy(Q):
    for sum in range(4,22):
        for upcard in range(1,11):
            for ace in range(2):
                print("sum:",sum,"upcard:",upcard,"ace:",ace,"Stick" if Q[sum][upcard][ace][0] > Q[sum][upcard][ace][1] else "Hit")

def make_table(Q):
    list = np.zeros([22,11])
    for sum in range(4,22):
        for upcard in range(1,11):
            list[sum][upcard] = (round(Q[sum][upcard][1][0],2),round(Q[sum][upcard][1][1],2)) #stick - hit

    with open('ace.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(list)

    list = np.zeros([22,11])
    for sum in range(4,22):
        for upcard in range(1,11):
            list[sum][upcard] = (round(Q[sum][upcard][0][0],2),round(Q[sum][upcard][0][1],2)) #stick - hit

    with open('not ace.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(list)


def deviationFromBS(Q):
    deviation = 0
    for sum in range(4,22):
        for upcard in range(1,11):
            for ace in range(2):
                #Determine Basic Strategy:
                if not ace:
                    if sum < 12: BS=1
                    if sum == 12:
                        if upcard in [4,5,6]: BS=0
                        else: BS=1
                    if sum in range(13,17):
                        if upcard in range(2,7): BS=0
                        else: BS=1
                    if sum >= 17: BS=0
                elif ace:
                    BS=0
                    if sum <= 17: BS=1
                    if sum == 18 and upcard > 8: BS=1
                if (Q[sum][upcard][ace][0] >= Q[sum][upcard][ace][1] and BS == 1) or (Q[sum][upcard][ace][0]<Q[sum][upcard][ace][1] and BS == 0) and not (sum<=11 and ace):
                    deviation+=1
                    #print("deviation from BS, sum:",sum,"upcard:",upcard,"ace:",ace,"Stick:",round(Q[sum][upcard][ace][0],2), "Hit:", round(Q[sum][upcard][ace][1],2),"BS:", BS)
    #print(f"deviations:{deviation}")
    return deviation

def basic_strategy(sum, dealer, ace):
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
    #soft totals
    elif ace:
        a=0
        if sum <= 17: a=1
        if sum == 18 and dealer > 8: a=1
    return a

#compute bs q values using SARSA
def compute_bs_values(y=.9):
    num_episodes = 10000000
    #Load the environment
    env = gym.make('Blackjack-v1')
    # Set learning parameters
    #create lists to contain total rewards and steps per episode
    #jList = []
    rList = []
    BS = np.zeros([32,11,2,2])
    times_visited = np.zeros([32,11,2,2])
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        env.state = (random.randint(4,21),random.randint(1,10),random.randint(0,1))
        rAll = 0
        d = False
        j = 0
        #The Q-Table learning algorithm
        while j < 99:
            j+=1
            sum = s[0]
            dealer=s[1]
            if s[2] == False: ace=0
            if s[2] == True: ace=1
            if j == 1:
                a = random.randint(0,1)
            else: a = basic_strategy(sum, dealer, ace)
            times_visited[sum,dealer,ace,a] +=1
            next_state,r,d,_ = env.step(a)
            next_sum = next_state[0]
            next_dealer = next_state[1]
            if next_state[2] == False: next_ace=0
            if next_state[2] == True: next_ace=1
            next_action = basic_strategy(next_sum, next_dealer, next_ace)
            #print("sum:",sum,"dealer:", dealer,"ace:", ace, "action:",a,"next_state:", next_state,"r:", r,"d:", d)
            alpha = 1/times_visited[sum,dealer,ace, a]
            if d == False:
                BS[sum,dealer,ace,a] = BS[sum,dealer,ace,a] + alpha*(r+y*BS[next_sum,next_dealer,next_ace,next_action] - BS[sum,dealer,ace,a])
            else: BS[sum,dealer,ace,a] = BS[sum,dealer,ace,a] + alpha*(r - BS[sum,dealer,ace,a])
            rAll += r
            s = next_state
            if d == True:
                break
        rList.append(rAll)
    printQ(times_visited)
    np.save('BS_values', BS)
    return BS

#compare learned q values to computed values from basic strategy to determine overestimation degree
def compute_overestimation(BS_values, Q):
    differences=[]
    for sum in range(4,22):
        for upcard in range(1,11):
            for ace in range(2):
                #Determine Basic Strategy:
                if not ace:
                    if sum < 12: BS=1
                    if sum == 12:
                        if upcard in [4,5,6]: BS=0
                        else: BS=1
                    if sum in range(13,17):
                        if upcard in range(2,7): BS=0
                        else: BS=1
                    if sum >= 17: BS=0
                elif ace:
                    BS=0
                    if sum <= 17: BS=1
                    if sum == 18 and upcard > 8: BS=1
                if not (ace and sum <= 11) and not sum == 21:
                    #differences.append(Q[sum][upcard][ace][0]-BS_values[sum][upcard][ace][0])
                    #print(f"{sum} {upcard} {ace} Stick Q:{Q[sum][upcard][ace][0]} BS:{BS_values[sum][upcard][ace][0]} diff:{differences[-1]}")
                    differences.append(Q[sum][upcard][ace][1]-BS_values[sum][upcard][ace][1])
                    #print(f"{sum} {upcard} {ace} Hit Q:{Q[sum][upcard][ace][1]} BS:{BS_values[sum][upcard][ace][1]} diff: {differences[-1]}")
    return np.mean(differences)


def compute_relative_overestimation(avg_reward, Q):
    freq = np.load("SSfreq.npy")
    sum_of_values = 0
    for sum in range(4,22):
        for upcard in range(1,11):
            for ace in range(2):
                if Q[sum][upcard][ace][0] >= Q[sum][upcard][ace][1]:
                    sum_of_values += Q[sum][upcard][ace][0]*freq[sum][upcard][ace]
                else:
                    sum_of_values += Q[sum][upcard][ace][1]*freq[sum][upcard][ace]
    difference = sum_of_values + 0.08
    return difference 