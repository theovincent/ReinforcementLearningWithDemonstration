import numpy as np
import matplotlib.pyplot as plt


class ReplayBuffer:
    def __init__(self, env, epsilon_decay, n_expert_samples=0, expert_policy=None):
        self.env = env
        self.epsilon_decay = epsilon_decay

        self.buffer_expert = []
        self.buffer_rl = []

        if n_expert_samples > 0:  # i.e with Demonstration
            self.collect_expert_samples(n_expert_samples, expert_policy, trajectory=False)

    def collect_expert_samples(self, n_samples, policy, trajectory=False):
        state = self.env.initial_state_distribution

        self.env.reset()
        terminal = False

        for idx_sample in range(n_samples):
            if not trajectory:
                state = np.random.choice(self.env._states)
                self.env.state = state
            elif terminal:  # Trajectory case
                self.env.reset()

            action = policy[state]

            next_state, reward, terminal, _ = self.env.step(action)
            next_action = policy[next_state]

            self.buffer_expert.append((state, action, reward, next_state, next_action))
            state = next_state

    def collect_rl_samples(self, n_samples, w, trajectory=False):
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

            self.buffer_rl.append((state, action, reward, next_state, next_action))
            state = next_state

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