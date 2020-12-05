import numpy as np


def argmax(q_values):
    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []

        if q_values[i] == top:
            ties.append(i)

    return np.random.choice(ties)


def max_action_count(action_values):
    count = 0

    for action_value in action_values:
        if action_value == np.max(action_values):
            count += 1

    return count
