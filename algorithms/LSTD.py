import numpy as np


class ReplayBuffer:
    def __init__(self, env, epsilon_decay, gamma, regularisor_bellmann):
        self.env = env
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.regularisor_bellmann = regularisor_bellmann

        self.size_feature = env.get_feature(0, 0).shape

        self.buffer = []

        self.A = None  # Grad matrix
        self.b = None  # Grad matrix

    def collect_samples(self, n_samples, w, trajectory):
        state = self.env.initial_state_distribution

        self.env.reset()
        terminal = False

        for idx_sample in range(n_samples):
            if not trajectory:
                state = np.random.choice(self.env._states)

                while self.env.is_terminal(state):
                    state = np.random.choice(self.env._states)

                self.env.state = state
            elif terminal:  # Trajectory case
                self.env.reset()

            # Policy improvement
            if np.random.random() < self.epsilon_decay(len(self.buffer)):
                action = np.random.choice(self.env._actions)
            else:
                action = np.argmax([self.env.get_feature(state, action) @ w for action in self.env._actions])

            next_state, reward, terminal, _ = self.env.step(action)
            next_action = np.argmax([self.env.get_feature(next_state, action) @ w for action in self.env._actions])

            self.buffer.append((state, action, reward, next_state, next_action))
            state = next_state

    def compute_grad_matrices(self):
        features = np.zeros((len(self.buffer), self.size_feature))
        next_features = np.zeros((len(self.buffer), self.size_feature))
        rewards = np.zeros(len(self.buffer))

        for idx_sample, (state, action, reward, next_state, next_action) in enumerate(self.buffer):
            features[idx_sample] = self.env.get_feature(state, action)
            next_features[idx_sample] = self.env.get_feature(next_state, next_action)
            rewards[idx_sample] = reward

        inverse_matrix = np.linalg.inv(
            features.T @ features + self.regularisor_bellmann * len(self.buffer) * np.eyes(self.size_feature)
        )

        self.A = np.eyes(self.size_feature) - self.gamma * inverse_matrix @ features.T @ next_features
        self.b = inverse_matrix @ features.T @ rewards


def lstd(env, n_samples_per_iteration, regularisor, regularisor_bellmann, trajectory=False, max_iteration=10, tol=1e-1):
    from algorithms.epsilon_greedy import epsilon_decay
    from simulators.grid_world import GAMMA

    replay_buffer = ReplayBuffer(env, epsilon_decay, GAMMA, regularisor_bellmann)

    w = np.random.randint(replay_buffer.size_feature)

    for iteration in range(max_iteration):
        # Exploration
        replay_buffer.collect_samples(n_samples_per_iteration, w, trajectory)
        replay_buffer.compute_grad_matrices()

        # Evaluation
        grad = float("inf") * np.ones(w.shape)

        while np.linalg.norm(grad) < tol:
            grad = np.zeros(w.shape)

            for sample in replay_buffer.buffer:
                feature = env.get_feature(sample[0])
                grad += feature.T @ (replay_buffer.A @ w + replay_buffer.b) @ replay_buffer.A.T @ feature

            grad = grad / len(replay_buffer.buffer) + regularisor * w

            w -= regularisor * grad

        # Improvement
        # Done when samples are collected.

    return [np.argmax([env.get_feature(state, action) @ w for action in env._actions]) for state in env._states]
