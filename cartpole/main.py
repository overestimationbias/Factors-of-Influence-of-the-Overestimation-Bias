#based on https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f
import gym
import copy
import torch
from torch.autograd import Variable
import random
import numpy as np

def random_search(env, episodes):
    """ Random search strategy implementation."""
    final_rewards = []
    for episode in range(episodes):
        done = False
        total = 0
        while not done:
            # Sample random actions
            action = env.action_space.sample()
            # Take action and extract results
            next_state, reward, done, _ = env.step(action)
            # Update reward
            total += reward
            if done:
                break
        # Add to the final_rewards reward
        final_rewards.append(total)
    return final_rewards

class DQN():
    ''' Deep Q Neural Network class. '''
    def __init__(self, state_dim, action_dim, lr=0.05):
            self.criterion = torch.nn.MSELoss()

            self.model = torch.nn.Sequential(
                            torch.nn.Linear(state_dim, 48),
                            torch.nn.ReLU(),
                            torch.nn.Linear(48, 92),
                            torch.nn.ReLU(),
                            torch.nn.Linear(92, action_dim)
                    )
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, state, y):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))
    
    def replay(self, memory, size, gamma=0.999):
        """New replay function"""
        #Try to improve replay speed
        if len(memory)>=size:
            batch = random.sample(memory,size)
            batch_t = list(map(list, zip(*batch))) #Transpose batch list
            states = batch_t[0]
            actions = batch_t[1]
            next_states = batch_t[2]
            rewards = batch_t[3]
            is_dones = batch_t[4]
        
            states = torch.Tensor(states)
            actions_tensor = torch.Tensor(actions)
            next_states = torch.Tensor(next_states)
            rewards = torch.Tensor(rewards)
            is_dones_tensor = torch.Tensor(is_dones)
        
            is_dones_indices = torch.where(is_dones_tensor==True)[0]
        
            all_q_values = self.model(states) # predicted q_values of all states
            all_q_values_next = self.model(next_states)
            #Update q values
            all_q_values[range(len(all_q_values)),actions]=rewards+gamma*torch.max(all_q_values_next, axis=1).values
            all_q_values[is_dones_indices.tolist(), actions_tensor[is_dones].tolist()]=rewards[is_dones_indices.tolist()]
        
            
            self.update(states.tolist(), all_q_values.tolist())

#Double Q Deep Learning Class.
class DQN_double(DQN):
    def __init__(self, state_dim, action_dim, lr):
        super().__init__(state_dim, action_dim, lr)
        self.target = copy.deepcopy(self.model)
        
    def target_predict(self, s):
        ''' Use target network to make predicitons.'''
        with torch.no_grad():
            return self.target(torch.Tensor(s))
        
    def target_update(self):
        ''' Update target network with the model weights.'''
        self.target.load_state_dict(self.model.state_dict())
        
    def replay(self, memory, size, gamma=1.0):
        ''' Add experience replay to the DQL network class.'''
        if len(memory) >= size:
            # Sample experiences from the agent's memory
            data = random.sample(memory, size)
            states = []
            targets = []
            # Extract datapoints from the data
            for state, action, next_state, reward, done in data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if done:
                    q_values[action] = reward
                else:
                    # The only difference between the simple replay is in this line
                    # It ensures that next q values are predicted with the target network.
                    q_values_next = self.target_predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                targets.append(q_values)
            self.update(states, targets)

#Self Correcting Q Deep Learning Class. 
class SCDQN(DQN):
    def __init__(self, state_dim, action_dim, lr):
        super().__init__(state_dim, action_dim, lr)
        self.target = copy.deepcopy(self.model)
        
    def target_predict(self, s):
        ''' Use target network to make predicitons.'''
        with torch.no_grad():
            return self.target(torch.Tensor(s))
        
    def target_update(self):
        ''' Update target network with the model weights.'''
        self.target.load_state_dict(self.model.state_dict())


    def replay(self, memory, size, gamma=1.0, beta=3):
        ''' Add experience replay to the DQL network class.'''
        if len(memory) >= size:
            # Sample experiences from the agent's memory
            data = random.sample(memory, size)
            states = []
            targets = []
            # Extract datapoints from the data
            for state, action, next_state, reward, done in data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if done:
                    q_values[action] = reward
                else:
                    # The only difference between the simple replay is in this line
                    # It ensures that next q values are predicted with the target network.
                    q_values_next_target = self.target_predict(next_state)
                    q_values_next_online = self.predict(next_state)
                    q_beta_values = q_values_next_target - beta*(q_values_next_target - q_values_next_online)
                    #print(q_beta_values)
                    ahat = torch.argmax(q_beta_values).item()
                    #print(ahat)
                    q_values[action] = reward + gamma * q_values_next_target[ahat]
                targets.append(q_values)
            self.update(states, targets)

#Universal learning function (the differences of algorithms is implemeneted in the NN classes)
def q_learning(env, model, episodes, gamma=0.999, 
               epsilon=1, replay=False, replay_size=512, double=False, 
               n_update=10, soft=False, verbose=True):
    """Deep Q Learning algorithm using the DQN. """
    final_rewards = []
    memory = []
    episode_i=0
    starting_states = []
    for episode in range(episodes):
        episode_i+=1
        if double and not soft:
            # Update target network every n_update steps
            if episode % n_update == 0:
                model.target_update()
        if double and soft:
            model.target_update()
        
        # Reset state
        state = env.reset()
        # predict and save overestimation to calculate overestimation 
        SS = DQN.predict(model, state)
        starting_states.append(max(SS.numpy()))
        done = False
        total = 0
        
        while not done:
            # Implement greedy search policy to explore the state space
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state)
                action = torch.argmax(q_values).item()
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Update total reward and memory
            total += reward
            memory.append((state, action, next_state, reward, done))
            q_values = model.predict(state).tolist()
             
            if done:
                if not replay:
                    q_values[action] = reward
                    # Update network weights
                    model.update(state, q_values)
                break

            if replay:
                # Update network weights using replay memory
                model.replay(memory, replay_size, gamma)

            else: 
                # Update network weights using the last step only
                q_values_next = model.predict(next_state)
                q_values[action] = reward + gamma * torch.max(q_values_next).item()
                model.update(state, q_values)

            state = next_state
        
        # Update epsilon
        epsilon = 1-(1/400)*episode
        final_rewards.append(total)
        
        if verbose:
            #print final starting state values overestimation
            if episode==episodes-1: 
                print(SS)
            print("episode: {}, total reward: {}".format(episode_i, total))
        
    return final_rewards, starting_states


env = gym.envs.make("CartPole-v0")
# Number of state observation variables (Cartpole: 4)
n_state = env.observation_space.shape[0]
# Number of actions (Cartpole: 2)
n_action = env.action_space.n
# Number of episodes
episodes = 500
# Number of experiments
experiments = 100

def run(env, lr = 0.0001, gamma=.999):
    # Run experiment
    starting_states = []
    rewards = []
    for i in range(experiments):
        print(i)
        simple_dqn = DQN(n_state, n_action, lr)
        results = q_learning(env, simple_dqn, episodes, gamma, verbose=True,replay=True, replay_size=512)
        rewards.append(results[0])
        starting_states.append(results[1])
    data=[rewards,starting_states]
    return data

def DDQLrun(env, lr = 0.0001, gamma=.999):
    # Run experiment
    starting_states = []
    rewards = []
    for i in range(experiments):
        print(i)
        dqn_double = DQN_double(n_state, n_action, lr)
        results =  q_learning(env, dqn_double, episodes, gamma=gamma, replay=True,verbose=True, double=True, replay_size=512)
        rewards.append(results[0])
        starting_states.append(results[1])
    data=[rewards,starting_states]
    return data

def SCQLrun(env, lr = 0.0001, gamma=.999):
    # Run experiment
    starting_states = []
    rewards = []
    for i in range(experiments):
        print(i)
        scqn = SCDQN(n_state, n_action, lr)
        results =  q_learning(env, scqn, episodes, gamma=gamma, replay=True,verbose=True, double=True, replay_size=512)
        rewards.append(results[0])
        starting_states.append(results[1])
    data=[rewards,starting_states]
    return data

#MAIN: run and save files with configuration here:
np.save("./data/acrobot_lr000001_y9", run(env,lr=0.0001,gamma=.97))

