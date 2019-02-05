import gym
import numpy as np
import random


env = gym.make('CartPole-v0').env
env.reset()


n_states = 500  # number of states
episodes = 100  # number of episodes
initial_lr = 1.0  # initial learning rate
min_lr = 0.005  # minimum learning rate
gamma = 0.99  # discount factor
max_steps = 300
epsilon = 0.05




q_table= np.zeros([env.observation_space.n,env.action_space.n])
print(q_table)

# for episode in range(1,100001):
#     state = env.reset()
#
#     done = False
#     alpha = max(min_lr, initial_lr*(gamma**(episode//10)))
#     epsilon = 1 - alpha
#
#     while not done:
#         if random.uniform(0,1)<epsilon:
#             action = env.action_space.sample()
#         else:
#             action = np.argmax(q_table[state])
#
#         next_state, reward, done, info = env.step(action)
#
#
#         next_max = np.max(q_table[next_state])
#
#         q_table[state,action] =  q_table[state,action] + alpha * ( reward + gamma * next_max - q_table[state,action])
#
#         state = next_state
#
#
#
# print('Training Finished..')