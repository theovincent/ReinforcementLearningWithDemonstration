import numpy as np


def optimise_u(env, samples, w, regularisor_bellmann, gamma):
    size_feature = env.get_feature(0, 0).shape[0]

    features = np.zeros((len(samples), size_feature))
    next_features = np.zeros((len(samples), size_feature))
    rewards = np.zeros(len(samples))

    for idx_sample, (state, action, reward, next_state, next_action) in enumerate(samples):
        features[idx_sample] = env.get_feature(state, action)
        next_features[idx_sample] = env.get_feature(next_state, next_action)
        rewards[idx_sample] = reward

    inverse_matrix = np.linalg.inv(features.T @ features + regularisor_bellmann * len(samples) * np.eye(size_feature))

    return inverse_matrix @ features.T @ (rewards + gamma * next_features @ w)


def optimise_w(env, samples, u, regularisor):
    size_feature = env.get_feature(0, 0).shape[0]

    features = np.zeros((len(samples), size_feature))

    for idx_sample, (state, action, _, _, _) in enumerate(samples):
        features[idx_sample] = env.get_feature(state, action)

    vector_ = features.T @ features @ u

    return np.linalg.inv(features.T @ features + regularisor * len(samples) * np.eye(size_feature)) @ vector_


def optimise_w_with_demonstration(env, samples, u, w, regularisor, regularisor_expert=0, expert_samples=None):
    size_feature = env.get_feature(0, 0).shape[0]

    features = np.zeros((len(samples), size_feature))

    for idx_sample, (state, action, _, _, _) in enumerate(samples):
        features[idx_sample] = env.get_feature(state, action)

    vector_ = features.T @ features @ u

    phi = 0
    for idx_sample, (state, action, _, _, _) in enumerate(expert_samples):
        Q_action = env.get_feature(state, action) @ w
        max_Q_other_action = -float("inf")
        best_other_action = None

        for other_action in range(env.A):
            if other_action == action:
                continue
            Q_other_action = env.get_feature(state, other_action) @ w

            if Q_other_action > max_Q_other_action:
                max_Q_other_action = Q_other_action
                best_other_action = other_action

        if 1 - Q_action + max_Q_other_action > 0:
            phi += env.get_feature(action, best_other_action) - env.get_feature(state, action)

    vector_ += regularisor_expert * phi

    return np.linalg.inv(features.T @ features + regularisor * len(samples) * np.eye(size_feature)) @ vector_
