# value iteration
import matplotlib.pyplot as plt
import numpy as np

v = np.array(101)
pHeads = 0.4

def valueIteration():
    # initialize the v table
    v = np.zeros(101)
    v[100] = 1.0 #terminal reward
    delta = 0

    while True:
        while True:
            delta = 0
            # loop through all possible states
            for s in range(1,100):
                vOld = v[s]
                best = -1
                # Bellman equation, loop through all possible actions maximizing
                for a in range(1,1 + min(s, (100-s))): #all possible bets
                    # for this action, two outcomes, heads or tails
                    # if heads get the money so next state is s+a
                    # if tails lose the money so next state is s-a
                    best = max(best, pHeads*v[s+a] + (1-pHeads)*v[s-a])
                # update the value of s with the new value
                v[s] = best
                delta = max(delta, abs(vOld - best))
            if delta < 1e-10:
                break
        # compute policy
        pi = np.zeros(101)
        # for each state
        for s in range(1,100):
            #old_action = pi[s]
            best = -1
            # try each action and do an argmax
            # note, to get the same graph as in the book, start with the largest
            # bets and use >= so policy will pick the smalest equivalent bet
            for a in range(min(s, (100-s)), 0, -1): #all possible bets
                # Note the round! So very similar propabilities look the same
                now = round(pHeads*v[s+a] + (1-pHeads)*v[s-a], 6)
                if now >= best:
                    best = now
                    bestAction = a
            pi[s] = bestAction
        policy_stable = True
        for s in range(1,100):
            previousAction = pi[s]
            best = -1
            # try each action and do an argmax
            # note, to get the same graph as in the book, start with the largest
            # bets and use >= so policy will pick the smalest equivalent bet
            for a in range(min(s, (100-s)), 0, -1): #all possible bets
                # Note the round! So very similar propabilities look the same
                new = round(pHeads*v[s+a] + (1-pHeads)*v[s-a], 6)
                if new > best:
                    best = new
                    bestAction = a
            pi[s] = bestAction
            if previousAction != pi[s]:
                policy_stable = False
        if policy_stable:
            break
        else:
            print("check")
            continue
    plt.plot(pi, drawstyle="steps")
    plt.show()

valueIteration()



