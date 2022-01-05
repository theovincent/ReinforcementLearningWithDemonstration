import numpy as np
import torch.nn as nn


class SmallNAC:
    def __init__(self, env, entropy_weight):
        self.env = env
        self.entropy_weight = entropy_weight

        self.net = nn.Sequential(
            nn.Linear(3, 50), nn.ReLU(), nn.Linear(50, 1)  # 2 for the state and 1 for the action  # 1 for Q(s, a)
        )

    def Q(self, state, action):
        (y, x) = self.env.index2coord[state]

        return self.net(x, y, action)

    def V(self, state):
        v_value = 0

        for action in self.env._actions:
            v_value += np.exp(self.Q(state, action) / self.entropy_weight)

        return self.entropy_weight * np.log(v_value)

    def pi(self, state):
        pi_actions = np.zeros((self.env.Na))

        for action in self.env._actions:
            pi_actions[action] = np.exp((self.Q(state, action) - self.V(state)) / self.entropy_weight)

        return np.random.choice(self.env._actions, p=pi_actions)
