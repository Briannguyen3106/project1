from __future__ import annotations

import numpy as np

from .common import epsilon_greedy_probabilities


class ExpectedSarsa:
    def __init__(self, state_count: int, action_count: int, alpha: float, gamma: float, seed: int):
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((state_count, action_count), dtype=float)

    def select_action(self, state: int, epsilon: float) -> int:
        probabilities = epsilon_greedy_probabilities(self.q_table[state], epsilon)
        return int(self.rng.choice(self.q_table.shape[1], p=probabilities))

    def update(self, state, action, reward, next_state, terminated, next_action=None, epsilon=0.0):
        if terminated:
            bootstrap = 0.0
        else:
            probabilities = epsilon_greedy_probabilities(self.q_table[next_state], epsilon)
            bootstrap = float(np.dot(probabilities, self.q_table[next_state]))
        target = reward + self.gamma * bootstrap
        error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * error
        return float(target), float(error)
