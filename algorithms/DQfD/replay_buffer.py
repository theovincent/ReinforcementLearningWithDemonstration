from random import choices
import numpy as np
import matplotlib.pyplot as plt


class Transition:
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

        self.next_transition = None

        self.td_loss = None


class ReplayBuffer:
    def __init__(
        self,
        env,
        epsilon_decay,
        n_expert_trajectories=0,
        expert_policy=None,
        n_step_td=10,
        prioritized_replay_rl_exploration=0.001,
        prioritized_replay_expert_exploration=1,
        prioritized_replay_exponent=0.4,
        importance_sampling=0.6,
    ):
        self.env = env
        self.epsilon_decay = epsilon_decay

        self.buffer_expert = []
        self.buffer_rl = []

        if n_expert_trajectories > 0:  # i.e with Demonstration
            self.collect_expert_trajectories(n_expert_trajectories, expert_policy)

        self.n_sted_td = n_step_td
        self.prioritized_replay_expert_exploration = prioritized_replay_expert_exploration
        self.prioritized_replay_rl_exploration = prioritized_replay_rl_exploration
        self.prioritized_replay_exponent = prioritized_replay_exponent
        self.importance_sampling = importance_sampling

        self.last_collected_transition = None
        self.last_sampled_transition = None

    def collect_expert_trajectories(self, n_trajectories, policy):
        for idx_sample in range(n_trajectories):
            self.env.reset()
            first_transition = True
            terminal = False

            while not terminal:
                action = policy[self.env.state]

                next_state, reward, terminal, _ = self.env.step(action)

                self.buffer_expert.append(Transition(self.env.state, action, reward, next_state))
                if not first_transition:
                    self.buffer_expert[-2].next_transition = self.buffer_expert[-1]

                first_transition = False

    def collect_rl_transition(self, Q):
        if self.last_collected_transition is None:
            self.env.reset()

        if np.random.random() < self.epsilon_decay(len(self.buffer_rl) + len(self.buffer_expert)):
            action = np.random.choice(self.env._actions)
        else:
            action = np.argmax([Q[self.env.state, action] for action in self.env._actions])

        next_state, reward, terminal, _ = self.env.step(action)

        self.buffer_rl.append(Transition(self.env.state, action, reward, next_state))
        if self.last_collected_transition is not None:
            self.last_collected_transition.next_transition = self.buffer_expert[-1]

        if not terminal:
            self.last_collected_transition = self.buffer_rl[-1]
        else:
            self.last_collected_transition = None

    def update_td_loss_last_sampled_transition(self, td_loss):
        self.last_sampled_transition.td_loss = td_loss

    def sample_a_transition(self):
        expert_probabilities = []
        for transition in self.buffer_expert:
            expert_probabilities.append(
                (np.abs(transition.td_loss) + self.prioritized_replay_expert_exploration)
                ** self.prioritized_replay_exponent
            )

        rl_probabilities = []
        for transition in self.buffer_rl:
            rl_probabilities.append(
                (np.abs(transition.td_loss) + self.prioritized_replay_rl_exploration)
                ** self.prioritized_replay_exponent
            )

        combined_probabilities = expert_probabilities + rl_probabilities
        combined_probabilities /= np.sum(combined_probabilities)

        idx_transition = choices(range(len(expert_probabilities + rl_probabilities)), combined_probabilities)

        if idx_transition < len(self.buffer_expert):
            return (
                self.buffer_expert[idx_transition],
                True,
                1 / (len(combined_probabilities) * combined_probabilities[idx_transition]) ** self.importance_sampling,
            )
        else:
            return (
                self.buffer_rl[idx_transition - len(self.buffer_expert)],
                False,
                1 / (len(combined_probabilities) * combined_probabilities[idx_transition]) ** self.importance_sampling,
            )

    def sample_n_transitions(self):
        first_transition, is_expert, weight = self.sample_a_transition()

        self.last_sampled_transition = first_transition
        transitions = [first_transition]
        n_transition = 0

        while n_transition < self.n_sted_td and not transitions[-1].next_transition is None:
            transitions.append(transitions[-1].next_transition)
            n_transition += 1

        return transitions, is_expert, weight

    def display_statistics_on_transitions(self):
        number_occurences = np.zeros(self.env.S)

        for transition in self.buffer_expert:
            number_occurences[transition.state] += 1

        for transition in self.buffer_rl:
            number_occurences[transition.state] += 1

        img = self.env.get_layout_img(number_occurences)
        plt.figure()
        plt.title("Statistics on occurences")
        plt.imshow(img)
        plt.show()
