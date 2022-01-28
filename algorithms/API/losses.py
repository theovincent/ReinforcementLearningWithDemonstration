import numpy as np


class LargeMarginExpertLoss:
    def __init__(self, env):
        self.env = env

    def __call__(self, w, state, action, return_grad=False):
        best_q_value = -float("inf")
        best_action = None

        for other_action in self.env._actions:
            if other_action == action:
                continue
            q_value = self.env.get_feature(state, other_action) @ w

            if q_value > best_q_value:
                best_q_value = q_value
                best_action = other_action

        if return_grad:
            return (1 - (self.env.get_feature(state, action) @ w - best_q_value) > 0) * (
                self.env.get_feature(state, best_action) - self.env.get_feature(state, action)
            )
        else:
            return max(1 - (self.env.get_feature(state, action) @ w - best_q_value), 0)


class PenalizationExpertLoss:
    def __init__(self, env, penality):
        self.env = env
        self.penality = penality

    def __call__(self, w, state, action, return_grad=False):
        best_penalized_q_value = -float("inf")
        best_penalized_action = None

        for other_action in self.env._actions:
            penalized_q_value = self.env.get_feature(state, other_action) @ w + self.penality * float(other_action != action)

            if penalized_q_value > best_penalized_q_value:
                best_penalized_q_value = penalized_q_value
                best_penalized_action = other_action

        if return_grad:
            return self.env.get_feature(state, best_penalized_action) - self.env.get_feature(state, action)
        else:
            return best_penalized_q_value - self.env.get_feature(state, action) @ w


class LossW:
    def __init__(self, env, regularisor, regularisor_expert, expert_loss_name, expert_penality):
        self.env = env
        self.regularisor = regularisor
        self.regularisor_expert = regularisor_expert

        if expert_loss_name == "large_margin":
            self.expert_loss = LargeMarginExpertLoss(env)
        elif expert_loss_name == "penalizer":
            self.expert_loss = PenalizationExpertLoss(env, expert_penality)
        else:
            self.expert_loss = None

    def __call__(self, w, samples_bellman, samples_expert, u):
        loss_bellman = 0
        for (state, action, _, _, _) in samples_bellman:
            loss_bellman += (self.env.get_feature(state, action) @ (w - u)) ** 2

        loss_bellman /= len(samples_bellman)

        loss_expert = 0
        for (state, action, _, _, _) in samples_expert:
            loss_expert += self.expert_loss(w, state, action)

        loss_expert /= len(samples_expert) if len(samples_expert) != 0 else 1

        return loss_bellman + self.regularisor * w @ w + self.regularisor_expert * loss_expert

    def compute_feature_matrix(self, samples_bellman):
        self.features = np.zeros((len(samples_bellman), self.env.dimensions))

        for idx_sample, (state, action, _, _, _) in enumerate(samples_bellman):
            self.features[idx_sample] = self.env.get_feature(state, action)

        self.features_T_features = self.features.T @ self.features / len(samples_bellman)

    def grad(self, w, samples_expert, u):
        grad_bellman = self.features_T_features @ (w - u)

        grad_expert = 0
        for (state, action, _, _, _) in samples_expert:
            grad_expert += self.expert_loss(w, state, action, return_grad=True)

        grad_expert /= len(samples_expert) if len(samples_expert) != 0 else 1

        return 2 * grad_bellman + 2 * self.regularisor * w + self.regularisor_expert * grad_expert
