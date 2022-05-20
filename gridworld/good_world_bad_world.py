import random 
import math
from operator import itemgetter
import numpy as np
import csv
from scipy.stats import sem

#states:
#  012
#  345
#  678
#
# 6 is the starting state, 2 is the goal state. Agents can move up,right,down,left
# They always move to the desired state successfully, moving into a wall results in not moving.

UP = 0
RIGHT = 1
DOWN = 2 
LEFT = 3 

TRUE = 1
FALSE = 0

DETERMINISTIC = 0
HIGH_VARIANCE_GAUSSIAN = 1 
LOW_VARIANCE_GAUSSIAN = 2
BERNOULLI = 3
REVERSED_BERNOULLI = 4

STANDARD = 0
STANDARD_AND_REWARDTABLE = 4
DOUBLE_Q = 6
STANDARD_AND_RUNNINGAVG = 7
SARSA = 8
EXPECTED_SARSA = 9

#new_state takes as input a state and an action and returns the new state 
def new_state(state, action):
    move = state
    if state in [2,11]:
        return random.choice([6,15])
    if action == UP and state not in [0,1,9,10]:
        move = (state-3)
    if action == RIGHT and state not in [5,8,14,17]:
        move = (state+1)
    if action == DOWN and state not in [6,7,8,15,16,17]:
        move = (state+3)
    if action == LEFT and state not in [0,3,6,9,12,15]:
        move = (state-1)
    #switch the world
    if np.random.rand() < 0.5:
        if state<=8: return move+9
        else: return move-9
    return move

# get_reward takes as input state and a reward function and returns the corresponding rewards: 
def get_reward(state):
    if state in [2,11]: return 5
    if state in [0,1,3,4,5,6,7,8]: return 10
    if state in [9,10,12,13,14,15,16,17]: return -12

#greedy choice takes a Q-table and a state and returns the action with the highest q values
def greedy_choice(Q,state):
    max_value = max(Q[state])
    return Q[state].index(max_value)

def expected_sarsa(Q, state, epsilon):
    sum = 0
    for i in range(4):
        if greedy_choice(Q,state) == i: sum += (1-(3/4)*epsilon)*Q[state][i]
        else: sum += 1/4*epsilon*Q[state][i]
    return sum

#solved_Q takes as input a Q table and prints if the Q table leads to a correct optimal path. 
def solved_Q(Q):
    state = 6
    moves = 0
    while state not in [2,11] and moves<=5:
        moves += 1
        state = new_state(state, greedy_choice(Q,state))
    if moves == 4:
        #print("Correct Q table")
        return 1
    else:
        #print("Incorrect Q table")
        return 0

# prints the Q table to show what path a greedy agent would take 
def show_moves(Q):
    for state in range(9):
        choice = greedy_choice(Q,state)
        if choice == 0: print("U",end ='')
        if choice == 1: print("R",end ='')
        if choice == 2: print("D",end ='')
        if choice == 3: print("L",end ='')
        if state in [2,5,8]: print(" ")

#prints a Q-Table
def printQ(Q):
    print("UP, RIGHT, DOWN, LEFT")
    for count, value in enumerate(Q):
        print (count, [round(num, 1) for num in value])

#MAIN
def gwbd_iteration(Q_function, discount_factor, dynamic_y,running_avg_range,dynamic_alpha, alpha):
    no_of_correct_qtables = 0
    rewards_over_time = [[0 for x in range(0,1000)] for y in range(0,10)]
    qmax_startingstate = [[0 for x in range(0,1000)] for y in range(0,10)]
    for experiment in range (1, 1000):
        reward_estimate = [0 for x in range(18)]
        running_avg_reward = [0 for x in range(18)]
        running_avg_reward1 = [0 for x in range(18)]
        running_avg_reward2 = [0 for x in range(18)]
        Q = [[0.0 for x in range(4)] for y in range(18)]
        Q1 = [[0.0 for x in range(4)] for y in range(18)]
        Q2 = [[0.0 for x in range(4)] for y in range(18)]
        Q_sum = [[0.0 for x in range(4)] for y in range(18)]
        times_visited = [0 for x in range(18)]
        times_visited_pairs = [[0 for x in range(4)] for y in range(18)]
        state = 6
        next_state = 6
        reward = 0
        for step in range(1,10000):
            state = next_state
            times_visited[state]+=1
            current_reward = get_reward(state)
            reward_estimate[state] = reward_estimate[state] + (current_reward-reward_estimate[state])/times_visited[state]
            running_avg_reward[state] = running_avg_reward[state] + (current_reward-running_avg_reward[state])/running_avg_range
            epsilon = 1/math.sqrt(times_visited[state])
            #Action Selection
            if Q_function == DOUBLE_Q:
                action = greedy_choice(Q_sum, state) if (random.random() > epsilon) else random.randint(0, 3)
            elif Q_function == SARSA:
                action = greedy_choice(Q,state)
            else:
                action = greedy_choice(Q, state) if (random.random() > epsilon) else random.randint(0, 3)
            next_state = new_state(state, action)
            #print(state, action, next_state, current_reward)
            times_visited_pairs[state][action]+=1
            if dynamic_alpha == TRUE: alpha = 1/times_visited_pairs[state][action]
            if dynamic_y == TRUE and step%1000==0 and discount_factor<0.99:
                discount_factor = 1-0.98*(1-discount_factor)
            if Q_function == DOUBLE_Q:
                if np.random.rand() < 0.5:
                    running_avg_reward1[state] = running_avg_reward1[state] + (current_reward-running_avg_reward1[state])/running_avg_range
                    if state in [2,11]: Q1[state][action] = Q1[state][action] + alpha*(current_reward - Q1[state][action])
                    elif times_visited[state]*2>running_avg_range: Q1[state][action] = Q1[state][action] + alpha*(current_reward+discount_factor*Q2[next_state][greedy_choice(Q1,next_state)] - Q1[state][action])
                else:
                    running_avg_reward2[state] = running_avg_reward2[state] + (current_reward-running_avg_reward2[state])/running_avg_range
                    if state in [2,11]: Q2[state][action] = Q2[state][action] + alpha*(current_reward - Q2[state][action])
                    elif times_visited[state]*2>running_avg_range: Q2[state][action] = Q2[state][action] + alpha*(current_reward+discount_factor*Q1[next_state][greedy_choice(Q2,next_state)] - Q2[state][action])
                Q_sum[state][action] = Q1[state][action] + Q2[state][action]
            #Q-value updates:
            if times_visited[state]>=running_avg_range: 
                if state in [2,11]: Q[state][action] = Q[state][action] + alpha*(current_reward - Q[state][action])
                else:
                    if Q_function == STANDARD: 
                        Q[state][action] = Q[state][action] + alpha*(current_reward+discount_factor*Q[next_state][greedy_choice(Q,next_state)] - Q[state][action])
                    if Q_function == STANDARD_AND_REWARDTABLE: 
                        Q[state][action] = Q[state][action] + alpha*(reward_estimate[state]+discount_factor*Q[next_state][greedy_choice(Q,next_state)] - Q[state][action])
                    if Q_function == STANDARD_AND_RUNNINGAVG: 
                        Q[state][action] = Q[state][action] + alpha*(running_avg_reward[state]+discount_factor*Q[next_state][greedy_choice(Q,next_state)] - Q[state][action]) 
                    if Q_function == SARSA: 
                        Q[state][action] = Q[state][action] + alpha*(current_reward+discount_factor*Q[next_state][greedy_choice(Q,next_state)] - Q[state][action])
                    if Q_function == EXPECTED_SARSA: 
                        Q[state][action] = Q[state][action] + alpha*(current_reward+discount_factor*expected_sarsa(Q,state,epsilon = 1/math.sqrt(times_visited[next_state]+1)) - Q[state][action])
            reward+=current_reward     
            if step%1000==999:
                rewards_over_time[int(step/1000)][experiment] = reward/step
                qmax_startingstate[int(step/1000)][experiment]= Q[6][greedy_choice(Q,6)] + Q_sum[6][greedy_choice(Q_sum,6)]
        no_of_correct_qtables += solved_Q(Q) + solved_Q(Q_sum)
        #printQ(Q_sum)
        #show_moves(Q_sum)
    print(Q_function, no_of_correct_qtables)
    return rewards_over_time, qmax_startingstate, no_of_correct_qtables





