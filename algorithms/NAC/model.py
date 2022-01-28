import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class SmallNAC(nn.Module):
    def __init__(self, env, entropy_weight):
        super(SmallNAC, self).__init__()
        self.env = env
        self.entropy_weight = entropy_weight

        self.net = nn.Sequential(nn.Linear(124, 1), nn.Sigmoid())

        self.net[0].weight.data = 0 * torch.ones(self.net[0].weight.data.size())
        self.net[0].bias.data = -3.4 * torch.ones(self.net[0].bias.data.size())

    def Q(self, state, action):
        return self.net(torch.tensor(self.env.get_feature(state, action), dtype=torch.float32))

    def V(self, state):
        v_value = 0

        for action in self.env._actions:
            v_value += torch.exp(self.Q(state, action) / self.entropy_weight)

        return self.entropy_weight * torch.log(v_value)

    def policy(self, state):
        return self.pi_distribution(state).sample()

    def pi_distribution(self, state):
        pi_actions = torch.zeros((self.env.Na))

        for action in self.env._actions:
            pi_actions[action] = torch.exp((self.Q(state, action) - self.V(state)) / self.entropy_weight)

        return Categorical(probs=pi_actions)

    def copy_to(self, target_model):
        target_model.load_state_dict(self.state_dict())
