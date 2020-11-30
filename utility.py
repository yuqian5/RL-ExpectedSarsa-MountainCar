import numpy as np


def argmax(q_values):
    """
    Takes in a list of q_values and returns the index of the item
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    top_value = float("-inf")
    ties = []

    for i in range(len(q_values)):
        # if a value in q_values is greater than the highest value update top and reset ties to zero
        # if a value is equal to top value add the index to ties
        # return a random selection from ties.
        if q_values[i] > top_value:
            top_value = q_values[i]
            ties.clear()
            ties.append(i)
        elif q_values[i] == top_value:
            ties.append(i)

    return np.random.choice(ties)


def max_action_count(action_values):
    count = 0

    for action_value in action_values:
        if action_value == max(action_values):
            count += 1

    return count
