import gym
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
from matplotlib import cm
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D

env = gym.make('Blackjack-v0')
env.reset()
observation_space = env.observation_space.spaces
action = 0
gamma = 0.99
obs, reward, complete, info = env.step(action)
state_space_size = (33, 12, 2)

# simple policy:
policy = np.zeros(state_space_size, dtype=int)

def observation_clean(obs):
    return obs[0], obs[1], int(obs[2])

def run_episode(policy, env=env):
    steps = []
    obs = observation_clean(env.reset())
    complete = False
    steps.append(((None, None)+ (obs, 0)))
    while not complete:
        action = policy[obs]
        observation_action = (obs, action)
        obs, reward, complete, info = env.step(action)
        obs = observation_clean(obs)
        steps.append(observation_action + (obs, int(reward)))
    return steps
    # list of tuples: (s, a, s', R)

N = np.zeros(state_space_size, dtype=int)
S = np.zeros(state_space_size)

for e in range(1000000):
    observations_reward = run_episode(policy)
    G = 0.
    for o0, a, o, r in reversed(observations_reward):
        G = r + gamma * G
        N[o] += 1
        S[o] += G

Gs = np.zeros(state_space_size)
Gs[N!=0] = S[N!=0]/N[N!=0]

def plot_non_usable_ace():
    X = np.arange(1, 11)
    Y = np.arange(4, 22)
    X, Y = np.meshgrid(X, Y)

    Z = Gs[4:22, 1:11, 0]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.hot,
                           linewidth=1)
    ax.set_ylabel("Player sum")
    ax.set_xlabel("Dealer show")
    ax.set_title("No useable Ace")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_usable_ace():
    X = np.arange(1, 11)
    Y = np.arange(12, 22)
    X, Y = np.meshgrid(X, Y)

    Z = Gs[12:22, 1:11, 1]
    fig = plt.figure(figsize=(10,8))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.hot,
                           linewidth=0, antialiased=False)
    ax.set_ylabel("Player sum")
    ax.set_xlabel("Dealer showing")
    ax.set_title("Useable Ace")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

returns = defaultdict(list)
nb_of_episodes = 500000

def run_episode_exploring_start(policy, env=env):
    steps = []
    observation = observation_clean(env.reset())
    done = False
    steps.append(((None, None)+ (observation, 0)))
    start = True
    while not done:
        if start:
            action = np.random.binomial(n=1, p=0.5)
            start = False
        else:
            action = policy[observation]
        observation_action = (observation, action)
        observation, reward, done, info = env.step(action)
        observation = observation_clean(observation)
        steps.append(observation_action + (observation, int(reward)))
    return steps

size = list(state_space_size) + [2]
Q = np.zeros(size)

def monte_carlo_optimal_policy(nb_of_episodes,
                               policy = np.random.binomial(n=1, p=0.5, size=state_space_size),
                               run_episode = run_episode_exploring_start):

    for i in range(nb_of_episodes):
        observations_reward = run_episode(policy)

        G = 0. # current return
        o_a = {} # map from states to (action, return)-Tuple
        for o0, a, o, r in reversed(observations_reward):
            G = r + gamma * G
            o_a[o0] = a, G
        for o, (a, G) in o_a.items():
            if o is not None:
                returns[(o, a)].append(G)
                re_mean = np.array(returns[(o, a)]).mean()
                Q[(o) + (a,)] = re_mean
                policy[o] = np.argmax(Q[(o)])
    return policy

policy = monte_carlo_optimal_policy(nb_of_episodes)

def plot_policy_useable_ace(policy):
    B = np.arange(4-0.5, 22-0.5, 0.2)
    A = np.arange(1-0.5, 11-0.5, 0.2)
    A, B = np.meshgrid(A, B)

    Po = scipy.ndimage.zoom(policy[4:22, 1:11, 0], 5)

    levels = range(-1,2)
    plt.figure(figsize=(7,6))
    CS = plt.contourf(A, B, Po, levels)
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('actions')
    plt.title('Policy for no useable ace')
    plt.xlabel("dealers showing")
    plt.ylabel("Players sum")
    _ = plt.xticks(range(1, 11))
    _ = plt.yticks(range(4, 22))
    plt.show()

def plot_policy_no_useable_ace(policy):
    B = np.arange(12-0.5, 22-0.5, 0.2)
    A = np.arange(1-0.5, 11-0.5, 0.2)
    A, B = np.meshgrid(A, B)

    Po = scipy.ndimage.zoom(policy[12:22, 1:11, 1], 5, mode='nearest')
    levels = range(-1,2)
    plt.figure(figsize=(7, 6))
    CS = plt.contourf(A, B, Po, levels)
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('actions')
    plt.title('Policy for useable ace')
    plt.xlabel("dealers showing")
    plt.ylabel("Players sum")
    _ = plt.xticks(range(1, 11))
    _ = plt.yticks(range(12, 22))
    plt.show()


plot_policy_useable_ace(policy)
# plot_policy_no_useable_ace(policy)


# plot_usable_ace()
# plot_non_usable_ace()
