import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
env = gym.make('Blackjack-v0')
np.random.seed(1234)


q = np.zeros((22, 12, 2, 2))
q[:20, :, :, 1] = 1
q[20:, :, :, 0] = 1

# Initialize state-action values
state_action_vals = np.zeros((q.shape))
# Initialize counts to 1 so that we don't divide by zero
state_action_counts = np.ones((q.shape))

scores = np.arange(12, 22)

def usable_ace(player):
    if 1 in player and sum(player) + 10 <= 21:
        return 1
    else:
        return 0

def hand_total(player):
    if usable_ace(player) == 0:
        return sum(player)
    return sum(player) + 10

def get_hand(total, ace):
    if total == 21:
        # Shuffle the order this is shown
        # to ensure dealer hands showing ace
        # are just as probable as showing 10
        # on 21
        p = np.random.rand()
        if p < 0.5:
            return [10, 1], 1
        else:
            return [1, 10], 1

    elif ace == 0:
        return [total - 10, 10], 0
    else:
        return [total - 11, 1], 1

for i in range(1000000):
    # Reset environment
    env.reset()

    # Randomly initialize starting point
    player_ace = np.random.choice([0, 1])
    player_sum = np.random.choice(scores)
    dealer_ace = np.random.choice([0, 1])
    dealer_sum = np.random.choice(scores)

    # Seed the OpenAI Gym environment with the random start
    env.player, player_ace = get_hand(player_sum, player_ace)
    env.dealer, dealer_ace = get_hand(dealer_sum, dealer_ace)

    # Randomize initial action
    action = np.random.choice([0, 1])
    # Log episode history
    state_action_history = []
    complete = False
    while complete == False:
        state_action_history.append([hand_total(env.player), env.dealer[0],
                                     usable_ace(env.player), action])
        # Take action
        s, reward, complete, _ = env.step(action)

        # Select action according to greedy policy
        action = np.argmax(q[np.min([s[0], 21]), s[1], int(s[2]), :])

    # Update state-action values after every complete game
    for j in state_action_history:
        state_action_vals[j[0], j[1], j[2], j[3]] += reward
        state_action_counts[j[0], j[1], j[2], j[3]] += 1

    q = state_action_vals / state_action_counts

"""Plotting Blackjack"""
# Without usable ace
fig = plt.figure(figsize=(10, 8))
ax = fig.gca(projection='3d')
player_range = np.arange(11, 22)
dealer_range = np.arange(1, 11)
X, Y = np.meshgrid(dealer_range, player_range)
Z = state_action_vals[4:22, 1:11, 0]
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=1,
                rstride=1, cstride=1)
ax.set_title("Without Ace")
ax.set_xlabel("Dealer Showing")
ax.set_ylabel("Player Hand")
ax.set_zlabel("State Value")
plt.show()


#
#
# # With usable ace
# fig = plt.figure(figsize=(12, 8))
# ax = fig.gca(projection='3d')
# player = np.arange(11, 22)
# dealer = np.arange(2, 12)
# X, Y = np.meshgrid(dealer_range, player_range)
# Z = state_action_vals[11:22,1:11,1].reshape(X.shape)
# ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=1,
#                 rstride=1, cstride=1)
# ax.set_title("With Ace")
# ax.set_xlabel("Dealer Showing")
# ax.set_ylabel("Player Hand")
# ax.set_zlabel("State Value")
# plt.show()