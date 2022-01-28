import numpy as np


class EpsilonDecay:
    def __init__(self, limit=0.2):
        self.limit = limit

    def __call__(self, n_samples):
        return max(1 / np.sqrt(n_samples + 1), self.limit)
