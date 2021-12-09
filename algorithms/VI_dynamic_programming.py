import numpy as np


def value_iteration(P, R, gamma, tol=1e-6):
    """
    Args:
        P: np.array
            transition matrix (NsxNaxNs)
        R: np.array
            reward matrix (NsxNa)
        gamma: float
            discount factor
        tol: float
            precision of the solution
    Return:
        Q: np.array
            final Q-function
        greedy_policy: np.array
            greedy policy
    """
    Ns, Na = R.shape
    Q_memory = np.zeros((Ns, Na))
    Q = R + gamma * P @ np.max(Q_memory, axis=1)

    while np.linalg.norm(Q - Q_memory) > tol:
        Q_memory = Q
        Q = R + gamma * P @ np.max(Q_memory, axis=1)

    return Q, np.argmax(Q, axis=1)
