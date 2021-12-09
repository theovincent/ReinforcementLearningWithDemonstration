import numpy as np


def epsilon_decay(n_samples):
    return 0.5 / (np.sqrt(n_samples) + 1)
