import matplotlib.pyplot as plt
import numpy as np


class ReplayBuffer:
    def __init__(self, env, n_expert_samples, expert_policy, batch_size):
        self.env = env

        self.buffer_expert = []
        self.buffer_rl = []

        if n_expert_samples > 0:  # i.e with Demonstration
            self.collect_expert_samples(n_expert_samples, expert_policy)

        self.batch_size = batch_size
        self.rl_state = self.env.initial_state_distribution
        self.terminal = False

    def collect_expert_samples(self, n_samples, policy):
        for idx_sample in range(n_samples):
            state = np.random.choice(self.env._states) if idx_sample != 0 else self.env.initial_state_distribution
            self.env.state = state

            action = policy[state]

            next_state, reward, _, _ = self.env.step(action)

            self.buffer_expert.append((state, action, reward, next_state))

    def collect_rl_sample(self, model):
        if self.terminal:
            self.rl_state = np.random.choice(self.env._states)
        self.env.state = self.rl_state

        action = model.policy(self.env.state).detach().numpy()

        next_state, reward, self.terminal, _ = self.env.step(action)

        self.buffer_rl.append((self.rl_state, action, reward, next_state))
        self.rl_state = next_state

    def get_batch(self, from_expert=True):
        if from_expert:
            return [
                self.buffer_expert[np.random.choice(range(len(self.buffer_expert)))] for _ in range(self.batch_size)
            ]
        else:
            return [self.buffer_rl[np.random.choice(range(len(self.buffer_rl)))] for _ in range(self.batch_size)]

    def display_statistics(self, from_expert):
        number_occurences = np.zeros(self.env.S)

        if from_expert:
            for (state, _, _, _) in self.buffer_expert:
                number_occurences[state] += 1
        else:
            for (state, _, _, _) in self.buffer_rl:
                number_occurences[state] += 1

        img = self.env.get_layout_img(number_occurences)
        plt.figure()
        plt.title("Statistics on occurences")
        plt.imshow(img)
        plt.show()
