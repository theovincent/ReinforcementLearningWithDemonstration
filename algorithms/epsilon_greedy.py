import numpy as np


def epsilon_decay(n_samples):
    return max(1 / (np.sqrt(n_samples) + 1), 0.2)
