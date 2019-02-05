import gym
import numpy as np
import gym
import itertools
import matplotlib
import numpy as np
import sys

# env_name = 'MountainCar-v0'
# env_name = 'Acrobot-v1'
env_name = 'CartPole-v1'

env = gym.make(env_name)

print("Action Set size :", env.action_space)
print("Observation set shape :", env.observation_space)
print("Highest state feature value :", env.observation_space.high)
print("Lowest state feature value:", env.observation_space.low)
print(env.observation_space.shape)

n_states = 500  # number of states
episodes = 100  # number of episodes
initial_lr = 1.0  # initial learning rate
min_lr = 0.005  # minimum learning rate
gamma = 0.99  # discount factor
max_steps = 300
epsilon = 0.05

env = env.unwrapped
env.seed(0)  # setting environment seed to reproduce same result
np.random.seed(0)  # setting numpy random number generation seed to reproduce same r

def discretize_range(lower_bound, upper_bound, num_bins):
    return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

num_discretization_bins = 7

state_bins = [
    # Cart position.
    discretize_range(-2.4, 2.4, num_discretization_bins),
    # Cart velocity.
    discretize_range(-3.0, 3.0, num_discretization_bins),
    # Pole angle.
    discretize_range(-0.5, 0.5, num_discretization_bins),
    # Tip velocity.
    discretize_range(-2.0, 2.0, num_discretization_bins)
]
max_bins = max(len(bin) for bin in state_bins)

def discretize_value(value, bins):
    return np.digitize(x=value, bins=bins)

def discretization(observation):
    # Discretize the observation features and reduce them to a single integer.
    state = sum(
        discretize_value(feature, state_bins[i]) * ((max_bins + 1) ** i)
        for i, feature in enumerate(observation)
    )
    return state



"""
Q table
rows are states but here state is 2-D pos,vel
columns are actions
therefore, Q - table would be 3-D"""
# q_table = np.zeros((n_states, n_states, env.action_space.n))
q_table = np.zeros(shape=(n_states, env.action_space.n))
total_steps = 0
for episode in range(episodes):
    obs = env.reset()
    total_reward = 0
    # decreasing learning rate alpha over time

    alpha = max(min_lr, initial_lr*(gamma**(episode//10)))

    epsilon = 1 - alpha

    steps = 0
    for i in range(max_steps):
        env.render()

        next_state = discretization(obs)


        #action for the current state using epsilon greedy
        if np.random.uniform(low=0, high=1) < epsilon:
            a = np.random.randint(0, 2)

        else:
            # a = np.argmax(q_table[position][velocity][pole_angle][pole_vel])
            a = np.argmax(q_table[next_state])
        obs, reward, terminate, _ = env.step(a)
        total_reward += abs(obs[0]+0.5)

        #q-table update
        current_state = next_state

        # q_table[position][velocity][pole_angle][pole_vel][a] = (1 - alpha) * q_table[position][velocity][pole_angle][pole_vel][a] + alpha * (reward + gamma * np.max(q_table[pos_][vel_][pa_][pvel_]))

        q_table[current_state, a] += min_lr * \
                                           (reward + gamma * max(q_table[next_state, :]) - q_table[current_state, a])
        steps += 1
        if terminate:
            break
    print("Episode {} completed with total reward {} in {} steps {} {}".format(episode+1, total_reward, steps, alpha, epsilon))

while True:
    #to hold the render at the last step when Car passes the flag
    env.render()


