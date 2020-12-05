import numpy as np
import tiles3 as tc
import utility


class MountainCarTileCoder:
    def __init__(self, iht_size=4096, num_tilings=8, num_tiles=8):
        self.iht = tc.IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles

    def get_tiles(self, position, velocity):
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
        np.random.seed()

        self.num_tilings = agent_info.get("num_tilings", 8)
        self.num_tiles = agent_info.get("num_tiles", 8)
        self.iht_size = agent_info.get("iht_size", 4096)
        self.epsilon = agent_info.get("epsilon", 0.0)
        self.gamma = agent_info.get("gamma", 1.0)
        self.alpha = agent_info.get("alpha", 0.1) / self.num_tilings
        self.initial_weights = agent_info.get("initial_weights", np.random.uniform(-0.001, 0, 1)[0])
        self.num_actions = agent_info.get("num_actions", 3)

        self.w = np.ones((self.num_actions, self.iht_size)) * self.initial_weights

        self.tc = MountainCarTileCoder(iht_size=self.iht_size,
                                       num_tilings=self.num_tilings,
                                       num_tiles=self.num_tiles)

    def select_action(self, tiles):

        action_values = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            action_values[action] = np.sum(self.w[action][tiles])

        if np.random.random() < self.epsilon:
            chosen_action = np.random.randint(0, self.num_actions)
        else:
            chosen_action = utility.argmax(action_values)

        return chosen_action, action_values[chosen_action]

    def agent_start(self, state):
        position, velocity = state

        active_tiles = self.tc.get_tiles(position, velocity)
        current_action, action_value = self.select_action(active_tiles)

        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        return self.last_action

    def agent_step(self, reward, state):
        position, velocity = state

        active_tiles = self.tc.get_tiles(position, velocity)
        current_action, action_value = self.select_action(active_tiles)

        # get all action values in current state
        action_values = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            action_values[action] = np.sum(self.w[action][active_tiles])

        # calculate pi(s,a)
        p_non_greedy = self.epsilon / self.num_actions
        p_greedy = ((1 - self.epsilon) / utility.max_action_count(action_values)) + p_non_greedy

        expected_action_value = 0

        max_action_value = np.max(action_values)

        for val in action_values:
            if val == max_action_value:
                expected_action_value += val * p_greedy
            else:
                expected_action_value += val * p_non_greedy

        last_action_value = np.sum(self.w[self.last_action][self.previous_tiles])
        self.w[self.last_action][self.previous_tiles] += self.alpha * (
                    reward + self.gamma * expected_action_value - last_action_value) * 1

        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        return self.last_action

    def agent_end(self, reward):
        last_action_value = np.sum(self.w[self.last_action][self.previous_tiles])
        self.w[self.last_action][self.previous_tiles] += self.alpha * (reward - last_action_value) * 1
