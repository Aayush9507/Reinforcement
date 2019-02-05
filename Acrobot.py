import gym
import numpy as np

# env_name = 'MountainCar-v0'
env_name = 'Acrobot-v1'
# env_name = 'CartPole-v0'

env = gym.make(env_name)

print("Action Set size :", env.action_space)
print("Observation set shape :", env.observation_space)
print("Highest state feature value :", env.observation_space.high)
print("Lowest state feature value:", env.observation_space.low)
print(env.observation_space.shape)

n_states = 40  # number of states
episodes = 100  # number of episodes
initial_lr = 1.0  # initial learning rate
min_lr = 0.005  # minimum learning rate
gamma = 0.99  # discount factor
max_steps = 300
epsilon = 0.05

env = env.unwrapped
env.seed(0)  # setting environment seed to reproduce same result
np.random.seed(0)  # setting numpy random number generation seed to reproduce same r


def discretization(env, obs):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_den = (env_high - env_low) / n_states

    pos_den = env_den[0]
    vel_den = env_den[1]

    pos_high = env_high[0]
    vel_high = env_high[1]

    pos_low = env_low[0]
    vel_low = env_low[1]

    pos_scaled = int((obs[0] - pos_low) / pos_den)  # converts to an integer value
    vel_scaled = int((obs[1] - vel_low) / vel_den)  # converts to an integer value
    return pos_scaled, vel_scaled


"""
Q table
rows are states but here state is 2-D pos,vel
columns are actions
therefore, Q - table would be 3-D"""
def render_game():
    q_table = np.zeros((n_states, n_states, env.action_space.n))

    total_steps = 0
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        # decreasing learning rate alpha over time

        alpha = max(min_lr, initial_lr*(gamma**(episode//10)))

        epsilon = 1 - alpha

        steps = 0
        while True:
            env.render()

            position, velocity = discretization(env, obs)


            #action for the current state using epsilon greedy
            if np.random.uniform(low=0, high=1) < epsilon:
                a = np.random.choice(env.action_space.n)
            else:
                a = np.argmax(q_table[position][velocity])
            obs, reward, terminate, _ = env.step(a)
            total_reward += abs(obs[0]+0.5)

            #q-table update
            pos_, vel_ = discretization(env, obs)
            q_table[position][velocity][a] = (1 - alpha) * q_table[position][velocity][a] + alpha * (reward + gamma * np.max(q_table[pos_][vel_]))
            steps += 1
            if terminate:
                break
        print("Episode {} completed with total reward {} in {} steps {} {}".format(episode+1, total_reward, steps, alpha, epsilon))

    while True:
        #to hold the render at the last step when Car passes the flag
        env.render()

render_game()