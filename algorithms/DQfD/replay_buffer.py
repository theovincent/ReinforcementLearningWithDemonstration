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


class ReplayBuffer:
    def __init__(
        self,
        env,
        epsilon_decay,
        n_expert_trajectories,
        expert_policy,
        n_step_td,
        prioritized_buffer,
        weight_occurencies,
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
        self.prioritized_buffer = prioritized_buffer
        self.weight_occurencies = weight_occurencies
        self.prioritized_replay_expert_exploration = prioritized_replay_expert_exploration
        self.prioritized_replay_rl_exploration = prioritized_replay_rl_exploration
        self.prioritized_replay_exponent = prioritized_replay_exponent
        self.importance_sampling = importance_sampling

        self.last_collected_transition = None
        self.last_sampled_transition = None

        self.intial_state_collected = False

        self.td_losses = [[None for _ in range(self.env.A)] for _ in range(self.env.S)]

    def collect_expert_trajectories(self, n_trajectories, policy):
        for idx_sample in range(n_trajectories):
            self.env.state = choices(self.env._states)[0]
            if self.env.state == self.env.initial_state_distribution:
                self.intial_state_collected = True

            first_transition = True
            terminal = False

            while not terminal:
                action = policy[self.env.state]
                state = self.env.state

                next_state, reward, terminal, _ = self.env.step(action)

                self.buffer_expert.append(Transition(state, action, reward, next_state))

                if not first_transition:
                    self.buffer_expert[-2].next_transition = self.buffer_expert[-1]

                first_transition = False

    def collect_rl_transition(self, Q):
        if self.last_collected_transition is None:
            self.env.state = choices(self.env._states)[0]
            if not self.intial_state_collected:
                self.env.state = self.env.initial_state_distribution
                self.intial_state_collected = True

        state = self.env.state

        if np.random.random() < self.epsilon_decay(len(self.buffer_rl) + len(self.buffer_expert)):
            action = np.random.choice(self.env._actions)
        else:
            action = np.argmax([Q[state, action] for action in self.env._actions])

        next_state, reward, terminal, _ = self.env.step(action)

        self.buffer_rl.append(Transition(state, action, reward, next_state))

        if self.last_collected_transition is not None:
            self.buffer_rl[-2].next_transition = self.buffer_rl[-1]

        if not terminal:
            self.last_collected_transition = self.buffer_rl[-1]
        else:
            self.last_collected_transition = None

    def sample_a_transition(self):
        if self.prioritized_buffer:
            list_td_losses = []
            number_occurences = np.zeros((self.env.S, self.env.A))

            for transition in self.buffer_expert:
                number_occurences[transition.state, transition.action] += 1
                if self.td_losses[transition.state][transition.action] is None:
                    list_td_losses.append(None)
                else:
                    list_td_losses.append(
                        (
                            np.abs(self.td_losses[transition.state][transition.action])
                            + self.prioritized_replay_expert_exploration
                        )
                        ** self.prioritized_replay_exponent
                    )

            for transition in self.buffer_rl:
                number_occurences[transition.state, transition.action] += 1
                if self.td_losses[transition.state][transition.action] is None:
                    list_td_losses.append(None)
                else:
                    list_td_losses.append(
                        (
                            np.abs(self.td_losses[transition.state][transition.action])
                            + self.prioritized_replay_rl_exploration
                        )
                        ** self.prioritized_replay_exponent
                    )

            td_losses_array = np.array(list_td_losses)
            max_td_losses = np.amax(td_losses_array, where=td_losses_array != None, initial=1)
            probabilities = np.where(
                td_losses_array == None, max_td_losses + 10, td_losses_array
            )  # +10 to force sampling the samples that were never visited.

            # Divide by the number of occurencies
            if self.weight_occurencies:
                for idx_transition, transition in enumerate(self.buffer_expert + self.buffer_rl):
                    probabilities[idx_transition] /= number_occurences[transition.state, transition.action]

            probabilities /= np.sum(probabilities)
        else:
            probabilities = np.ones(len(self.buffer_expert) + len(self.buffer_rl)) / (
                len(self.buffer_expert) + len(self.buffer_rl)
            )

        idx_transition = choices(range(len(probabilities)), probabilities)[0]

        if idx_transition < len(self.buffer_expert):
            return (
                self.buffer_expert[idx_transition],
                True,
                1 / (len(probabilities) * probabilities[idx_transition]) ** self.importance_sampling,
            )
        else:
            return (
                self.buffer_rl[idx_transition - len(self.buffer_expert)],
                False,
                1 / (len(probabilities) * probabilities[idx_transition]) ** self.importance_sampling,
            )

    def sample_n_transitions(self):
        first_transition, is_expert, weight = self.sample_a_transition()

        self.last_sampled_transition = first_transition
        transitions = [first_transition]
        n_transition = 1

        while n_transition < self.n_sted_td and not transitions[-1].next_transition is None:
            transitions.append(transitions[-1].next_transition)
            n_transition += 1

        return transitions, is_expert, weight

    def display_statistics(self, transitions):
        number_occurences = np.zeros(self.env.S)

        for transition in transitions:
            number_occurences[transition.state] += 1

        img = self.env.get_layout_img(number_occurences)
        plt.figure()
        plt.title("Statistics on occurences")
        plt.imshow(img)
        plt.show()

    def display_rl_statistics(self):
        number_occurences = np.zeros(self.env.S)

        for transition in self.buffer_rl:
            number_occurences[transition.state] += 1

        print("Max occurences", np.max(number_occurences))
        img = self.env.get_layout_img(number_occurences)
        plt.figure()
        plt.title("Statistics on rl occurences")
        plt.imshow(img)
        plt.show()
