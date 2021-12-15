def get_bellman_error(env, buffer_, w, u, regularisor_bellmann, gamma):
    error = 0
    for state, action, reward, next_state, next_action in buffer_:
        error += (
            env.get_feature(state, action).T @ u - (reward + gamma * env.get_feature(next_state, next_action).T @ w)
        ) ** 2

    return error / len(buffer_) + regularisor_bellmann * u.T @ u


def get_minimisation_error(env, buffer_, w, u, regularisor):
    error = 0
    for state, action, _, _, _ in buffer_:
        feature = env.get_feature(state, action)
        error += (feature.T @ w - feature.T @ u) ** 2

    return error / len(buffer_) + regularisor * w.T @ w
