import random 
import math
import numpy as np

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

DETERMINISTIC = 0
HIGH_VARIANCE_GAUSSIAN = 1 
LOW_VARIANCE_GAUSSIAN = 2
BERNOULLI = 3
REVERSED_BERNOULLI = 4

STANDARD = 0
DOUBLE_Q = 1
STANDARD_AND_REWARDTABLE = 2
STANDARD_AND_RUNNINGAVG = 3
DOUBLE_Q_RUNNINGAVG = 4
SARSA = 5
EXPECTED_SARSA = 6
SELF_CORRECTING = 7 

#new_state takes as input a state and an action and returns the new state 
def new_state(state, action):
    if state == 2:
        return 6
    if action == UP and state not in [0,1]:
        return (state-3)
    if action == RIGHT and state not in [5,8]:
        return (state+1)
    if action == DOWN and state not in [6,7,8]:
        return (state+3)
    if action == LEFT and state not in [0,3,6]:
        return (state-1)
    return state


# get_reward takes as input state and a reward function and returns the corresponding rewards: 
def get_reward(state, reward_function):
    if state != 2:
        if reward_function == DETERMINISTIC or reward_function == REVERSED_BERNOULLI: 
            return -1
        if reward_function == HIGH_VARIANCE_GAUSSIAN: 
            return np.random.normal(-1, 5)
        if reward_function == LOW_VARIANCE_GAUSSIAN: 
            return np.random.normal(-1, 1)
        if reward_function == BERNOULLI: 
            return random.choice([10,-12])
    else: 
        if reward_function == REVERSED_BERNOULLI: 
            return random.choice([105,-105])
        else: 
            return 5

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
    while state != 2 and moves<=5:
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
def iteration(Q_function, reward_function=BERNOULLI, discount_factor=0.95, dynamic_y=False,running_avg_range=1,dynamic_alpha=True,alpha=0.1, beta=4):
    no_of_correct_qtables = 0
    rewards_over_time = [[0 for x in range(0,1000)] for y in range(0,10000)]
    qmax_startingstate = [[0 for x in range(0,1000)] for y in range(0,10000)]
    for experiment in range (1, 1000):
        reward_estimate = [0 for x in range(9)]
        running_avg_reward = [0 for x in range(9)]
        running_avg_reward1 = [0 for x in range(9)]
        running_avg_reward2 = [0 for x in range(9)]
        Q = [[0.0 for x in range(4)] for y in range(9)]
        Q1 = [[0.0 for x in range(4)] for y in range(9)]
        Q2 = [[0.0 for x in range(4)] for y in range(9)]
        Q_sum = [[0.0 for x in range(4)] for y in range(9)]
        Q_BETA = [[0.0 for x in range(4)] for y in range(9)]
        Q_PREVIOUS = [[0.0 for x in range(4)] for y in range(9)]
        times_visited = [0 for x in range(9)]
        times_visited_pairs = [[0 for x in range(4)] for y in range(9)]
        state = 6
        next_state = 6
        reward = 0
        for step in range(1,10000):
            state = next_state
            times_visited[state]+=1
            current_reward = get_reward(state, reward_function)
            reward_estimate[state] = reward_estimate[state] + (current_reward-reward_estimate[state])/times_visited[state]
            running_avg_reward[state] = running_avg_reward[state] + (current_reward-running_avg_reward[state])/running_avg_range
            epsilon = 1/math.sqrt(times_visited[state])
            #Action Selection
            if Q_function in [DOUBLE_Q, DOUBLE_Q_RUNNINGAVG]:
                action = greedy_choice(Q_sum, state) if (random.random() > epsilon) else random.randint(0, 3)
            elif Q_function == SARSA:
                action = greedy_choice(Q,state)
            else:
                action = greedy_choice(Q, state) if (random.random() > epsilon) else random.randint(0, 3)
            #visiting new state
            next_state = new_state(state, action)
            times_visited_pairs[state][action]+=1
            if dynamic_alpha == True: alpha = 1/times_visited_pairs[state][action]
            if dynamic_y == True and step%1000==0 and discount_factor<0.99:
                discount_factor = 0.6-0.9*(0.6-discount_factor)
            #Q-value updates
            if Q_function == DOUBLE_Q_RUNNINGAVG:
                if np.random.rand() < 0.5:
                    running_avg_reward1[state] = running_avg_reward1[state] + (current_reward-running_avg_reward1[state])/running_avg_range
                    if state == 2: Q1[state][action] = Q1[state][action] + alpha*(current_reward - Q1[state][action])
                    elif times_visited[state]*2>running_avg_range: Q1[state][action] = Q1[state][action] + alpha*(running_avg_reward1[state]+discount_factor*Q2[next_state][greedy_choice(Q1,next_state)] - Q1[state][action])
                else:
                    running_avg_reward2[state] = running_avg_reward2[state] + (current_reward-running_avg_reward2[state])/running_avg_range
                    if state == 2: Q2[state][action] = Q2[state][action] + alpha*(current_reward - Q2[state][action])
                    elif times_visited[state]*2>running_avg_range: Q2[state][action] = Q2[state][action] + alpha*(running_avg_reward2[state]+discount_factor*Q1[next_state][greedy_choice(Q2,next_state)] - Q2[state][action])
                Q_sum[state][action] = Q1[state][action] + Q2[state][action]
            if Q_function == DOUBLE_Q:
                if np.random.rand() < 0.5:
                    if state == 2: Q1[state][action] = Q1[state][action] + alpha*(current_reward - Q1[state][action])
                    else: Q1[state][action] = Q1[state][action] + alpha*(current_reward+discount_factor*Q2[next_state][greedy_choice(Q1,next_state)] - Q1[state][action])
                else:
                    if state == 2: Q2[state][action] = Q2[state][action] + alpha*(current_reward - Q2[state][action])
                    else: Q2[state][action] = Q2[state][action] + alpha*(current_reward+discount_factor*Q1[next_state][greedy_choice(Q2,next_state)] - Q2[state][action])
                Q_sum[state][action] = Q1[state][action] + Q2[state][action]
            if Q_function == SELF_CORRECTING:
                Q_BETA[next_state][action] = Q[next_state][action] - beta*(Q[next_state][action]-Q_PREVIOUS[next_state][action])
                ahat = greedy_choice(Q_BETA,next_state)
                Q_PREVIOUS[state][action] = Q[state][action]
                if state == 2: Q[state][action] = Q[state][action] + alpha*(current_reward - Q[state][action])
                else: Q[state][action] = Q[state][action] + alpha*(current_reward+discount_factor*Q[next_state][ahat] - Q[state][action])
            if times_visited[state]>=running_avg_range: 
                if state == 2: Q[state][action] = Q[state][action] + alpha*(current_reward - Q[state][action])
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
            rewards_over_time[step][experiment] = reward/step
            qmax_startingstate[step][experiment]= Q[6][greedy_choice(Q,6)] + Q_sum[6][greedy_choice(Q_sum,6)]
        no_of_correct_qtables += solved_Q(Q) + solved_Q(Q_sum)
    print(no_of_correct_qtables)
    return rewards_over_time, qmax_startingstate, no_of_correct_qtables
