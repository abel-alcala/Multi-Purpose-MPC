from collections import defaultdict

import numpy as np
from RL_Env import TrackingEnv

# Discrete action set for Q-learning, each action [velocity, steering_angle]
ACTIONS = np.array([
    [0.20, -0.45],
    [0.20, 0.00],
    [0.20, 0.45],

    [0.50, -0.35],
    [0.50, 0.00],
    [0.50, 0.35],

    [0.80, -0.25],
    [0.80, 0.00],
    [0.80, 0.25],
], dtype=np.float32)

class QLearningAgent:
    def __init__(
            self,
            n_actions,
            alpha=0.15, # learning rate, how strongly new info overrides old
            gamma=0.98, # how much agent cares about future reward
            epsilon=1.0, # exploration or exploitation

    ):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma


        # agent sees a new state first time, create a zero filled list of Q-values for all actions
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))

    # choose_action() - Decide to choose best known action or act randomly
    def choose_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        return int(np.argmax(self.q_table[state]))

    # update() - finds best move that will maximize the reward
    def update(self, state, action_id, reward, next_state, done):
        current_q = self.q_table[state][action_id]

        if done:
            target_q = reward
        else:
            # new value = reward + importance of future reward * best future reward from next state
            target_q = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action_id] += self.alpha * (target_q - current_q)
