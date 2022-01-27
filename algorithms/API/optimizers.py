import numpy as np


def optimise_u(env, samples, w, regularisor_bellmann):
    features = np.zeros((len(samples), env.dimensions))
    next_features = np.zeros((len(samples), env.dimensions))
    rewards = np.zeros(len(samples))

    for idx_sample, (state, action, reward, next_state, next_action) in enumerate(samples):
        features[idx_sample] = env.get_feature(state, action)
        next_features[idx_sample] = env.get_feature(next_state, next_action)
        rewards[idx_sample] = reward

    inverse_matrix = np.linalg.inv(features.T @ features + regularisor_bellmann * len(samples) * np.eye(env.dimensions))

    return inverse_matrix @ features.T @ (rewards + env.gamma * next_features @ w)


def optimise_w(loss_w, w, samples_bellman, samples_expert, u, learning_rate):
    grad_w = float("inf")
    count = 0

    loss_w.compute_feature_matrix(samples_bellman)

    while np.linalg.norm(grad_w) > 1e-6 and count < 1000:
        grad_w = loss_w.grad(w, samples_expert, u)
        w += -learning_rate * grad_w
        count += 1

    if count == 1000:
        print("! Warning ! Stopped before convergence")
        print("Grad norm", np.linalg.norm(grad_w))

    return w


# For debugging
def optimise_w_exact(env, samples, u, regularisor):
    features = np.zeros((len(samples), env.dimensions))

    for idx_sample, (state, action, _, _, _) in enumerate(samples):
        features[idx_sample] = env.get_feature(state, action)

    inverse_matrix = np.linalg.inv(features.T @ features + regularisor * len(samples) * np.eye(env.dimensions))

    return inverse_matrix @ features.T @ features @ u
