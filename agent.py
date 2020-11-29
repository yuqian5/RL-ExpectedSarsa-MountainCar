import numpy as np
import itertools
import tiles3 as tc
import time
import utility

class MountainCarTileCoder:
    def __init__(self, iht_size=4096, num_tilings=8, num_tiles=8):
        self.iht = tc.IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles

    def get_tiles(self, position, velocity):
        """
        Takes in a position and velocity from the mountaincar environment
        and returns a numpy array of active tiles.

        Arguments:
        position -- float, the position of the agent between -1.2 and 0.5
        velocity -- float, the velocity of the agent between -0.07 and 0.07
        returns:
        tiles - np.array, active tiles
        """
        position_scaled = (position + 1.2) / (0.5 + 1.2) * self.num_tiles
        velocity_scaled = (velocity + 0.07) / (0.07 + 0.07) * self.num_tiles

        # get the tiles using tc.tiles, with self.iht, self.num_tilings and [scaled position, scaled velocity]
        # nothing to implment here
        tiles = tc.tiles(self.iht, self.num_tilings, [position_scaled, velocity_scaled])

        return np.array(tiles)


class ExpectedSarsaAgent:
    def __init__(self):
        self.last_action = None
        self.last_state = None
        self.epsilon = None
        self.gamma = None
        self.iht_size = None
        self.w = None
        self.alpha = None
        self.num_tilings = None
        self.num_tiles = None
        self.mctc = None
        self.initial_weights = None
        self.num_actions = None
        self.previous_tiles = None
        self.tc = None

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        self.num_tilings = agent_info.get("num_tilings", 8)
        self.num_tiles = agent_info.get("num_tiles", 8)
        self.iht_size = agent_info.get("iht_size", 4096)
        self.epsilon = agent_info.get("epsilon", 0.0)
        self.gamma = agent_info.get("gamma", 1.0)
        self.alpha = agent_info.get("alpha", 0.1) / self.num_tilings
        self.initial_weights = agent_info.get("initial_weights", 0.0)
        self.num_actions = agent_info.get("num_actions", 3)

        # We initialize self.w to three times the iht_size. Recall this is because
        # we need to have one set of weights for each action.
        self.w = np.ones((self.num_actions, self.iht_size)) * self.initial_weights

        # We initialize self.mctc to the mountaincar verions of the
        # tile coder that we created
        self.tc = MountainCarTileCoder(iht_size=self.iht_size,
                                       num_tilings=self.num_tilings,
                                       num_tiles=self.num_tiles)

    def select_action(self, tiles):
        """
        Selects an action using epsilon greedy
        Args:
        tiles - np.array, an array of active tiles
        Returns:
        (chosen_action, action_value) - (int, float), tuple of the chosen action
                                        and it's value
        """
        action_values = []
        chosen_action = None

        for i in range(self.num_actions):
            action_values.append(self.w[i][tiles].sum())

        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.num_actions)
        else:
            chosen_action = utility.argmax(action_values)

        return chosen_action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        position, velocity = state

        active_tiles = self.tc.get_tiles(position, velocity)
        current_action = self.select_action(active_tiles)

        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        position, velocity = state

        active_tiles = self.tc.get_tiles(position, velocity)
        current_action = self.select_action(active_tiles)

        # calculate probabilities for greedy and non greedy actions, used to calculate expected_action_value
        action_values = []
        for i in range(self.num_actions):
            action_values.append(self.w[i][active_tiles].sum())
        p_non_greedy = self.epsilon / self.num_actions
        p_greedy = ((1 - self.epsilon) / utility.max_action_count(action_values)) + p_non_greedy
        max_action_value = max(action_values)

        # calculate expected_action_value
        expected_action_value = 0
        for action_value in action_values:
            if action_value == max_action_value:
                expected_action_value += p_greedy * action_value
            else:
                expected_action_value += p_non_greedy * action_value

        # calculate last_action_value
        last_action_value = self.w[self.last_action][self.previous_tiles].sum()

        delta = reward + self.gamma * expected_action_value - last_action_value
        grad = np.zeros_like(self.w)
        grad[self.last_action][self.previous_tiles] = 1
        self.w += self.alpha * delta * grad

        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        last_action_value = self.w[self.last_action][self.previous_tiles].sum()

        grad = np.zeros_like(self.w)
        grad[self.last_action][self.previous_tiles] = 1

        self.w += self.alpha * (reward - last_action_value) * grad

    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        pass

    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        pass