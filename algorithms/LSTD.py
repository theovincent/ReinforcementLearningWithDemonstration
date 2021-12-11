import numpy as np
import matplotlib.pyplot as plt


def optimise_u(env, buffer_, w, regularisor_bellmann, gamma):
    size_feature = env.get_feature(0, 0).shape[0]

    features = np.zeros((len(buffer_), size_feature))
    next_features = np.zeros((len(buffer_), size_feature))
    rewards = np.zeros(len(buffer_))

    for idx_sample, (state, action, reward, next_state, next_action) in enumerate(buffer_):
        features[idx_sample] = env.get_feature(state, action)
        next_features[idx_sample] = env.get_feature(next_state, next_action)
        rewards[idx_sample] = reward

    inverse_matrix = np.linalg.inv(features.T @ features + regularisor_bellmann * len(buffer_) * np.eye(size_feature))

    return inverse_matrix @ features.T @ (rewards + gamma * next_features @ w)


def optimise_w(env, buffer_, u, regularisor):
    size_feature = env.get_feature(0, 0).shape[0]

    features = np.zeros((len(buffer_), size_feature))

    for idx_sample, (state, action, _, _, _) in enumerate(buffer_):
        features[idx_sample] = env.get_feature(state, action)

    return np.linalg.inv(features.T @ features + regularisor * np.eye(size_feature)) @ features.T @ features @ u


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


class ReplayBuffer:
    def __init__(self, env, epsilon_decay, gamma, regularisor_bellmann):
        self.env = env
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.regularisor_bellmann = regularisor_bellmann

        self.buffer = []

    def collect_samples(self, n_samples, w, trajectory):
        state = self.env.initial_state_distribution

        self.env.reset()
        terminal = False

        for idx_sample in range(n_samples):
            if not trajectory:
                state = np.random.choice(self.env._states)
                self.env.state = state
            elif terminal:  # Trajectory case
                self.env.reset()

            # Policy improvement
            if np.random.random() < self.epsilon_decay(n_samples):
                action = np.random.choice(self.env._actions)
            else:
                action = np.argmax([self.env.get_feature(state, action) @ w for action in self.env._actions])

            next_state, reward, terminal, _ = self.env.step(action)
            next_action = np.argmax([self.env.get_feature(next_state, action) @ w for action in self.env._actions])

            self.buffer.append((state, action, reward, next_state, next_action))
            state = next_state

    def display_statistics_on_samples(self):
        plt.figure()
        number_occurences = np.zeros(self.env.S)

        for (state, _, _, _, _) in self.buffer:
            number_occurences[state] += 1

        img = self.env.get_layout_img(number_occurences)
        # print("Number of sample on terminal state:", number_occurences[5])
        # print("Least amount of occurences: ", np.min(number_occurences))
        plt.title("Statistics on occurences")
        plt.imshow(img)
        plt.show()


def lstd(
    env,
    n_samples_per_iteration,
    regularisor,
    regularisor_bellmann,
    max_iteration=10,
    show_policy=False,
    show_value_function=False,
    show_statistics=False,
):
    from algorithms.epsilon_greedy import epsilon_decay
    from simulators.grid_world.displays import display_policy, display_value_function

    from simulators.grid_world import GAMMA

    replay_buffer = ReplayBuffer(env, epsilon_decay, GAMMA, regularisor_bellmann)
    size_feature = env.get_feature(0, 0).shape[0]

    w = np.zeros(size_feature)

    for iteration in range(max_iteration):
        # Exploration
        replay_buffer.collect_samples(n_samples_per_iteration, w, False)
        if show_statistics:
            replay_buffer.display_statistics_on_samples()

        # Evaluation
        u = optimise_u(env, replay_buffer.buffer, w, regularisor_bellmann, GAMMA)
        w = optimise_w(env, replay_buffer.buffer, u, regularisor)

        # Improvement
        # Done when samples are collected.

        if show_value_function:
            display_value_function(env, w)
        if show_policy:
            display_policy(env, w)

    Q = np.zeros((env.S, env.A))

    for state in env._states:
        for action in env._actions:
            Q[state, action] = env.get_feature(state, action) @ w

    return Q, np.argmax(Q, axis=1), replay_buffer.buffer
