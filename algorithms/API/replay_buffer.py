import numpy as np
import matplotlib.pyplot as plt


class ReplayBuffer:
    def __init__(self, env, epsilon_decay, n_expert_samples=0, expert_policy=None):
        self.env = env
        self.epsilon_decay = epsilon_decay

        self.buffer_expert = []
        self.buffer_rl = []

        if n_expert_samples > 0:  # i.e with Demonstration
            self.collect_expert_samples(n_expert_samples, expert_policy)

    def collect_expert_samples(self, n_samples, policy):
        self.env.reset()
        state = self.env.state

        for idx_sample in range(n_samples):
            if idx_sample > 0:
                state = np.random.choice(self.env._states)
                self.env.state = state

            action = policy[state]

            next_state, reward, _, _ = self.env.step(action)
            next_action = policy[next_state]

            self.buffer_expert.append((state, action, reward, next_state, next_action))

    def collect_rl_samples(self, n_samples, w, iteration):
        # Reset buffer rl to have strict on policy learning
        self.buffer_rl = []

        self.env.reset()
        state = self.env.state

        for idx_sample in range(n_samples):
            if idx_sample > 0:
                state = np.random.choice(self.env._states)
                self.env.state = state

            # Policy improvement
            if np.random.random() < self.epsilon_decay(2 * iteration):
                action = np.random.choice(self.env._actions)
            else:
                action = np.argmax([self.env.get_feature(state, action) @ w for action in self.env._actions])

            next_state, reward, _, _ = self.env.step(action)
            next_action = np.argmax([self.env.get_feature(next_state, action) @ w for action in self.env._actions])

            self.buffer_rl.append((state, action, reward, next_state, next_action))

    def display_statistics_on_samples(self):
        plt.figure()
        number_occurences = np.zeros(self.env.S)

        for (state, _, _, _, _) in self.buffer_expert:
            number_occurences[state] += 1

        for (state, _, _, _, _) in self.buffer_rl:
            number_occurences[state] += 1

        img = self.env.get_layout_img(number_occurences)
        plt.title("Statistics on occurences")
        plt.imshow(img)
        plt.show()
