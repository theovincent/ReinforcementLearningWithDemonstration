import torch


def get_q_learning_loss(env, batch, model, target_model):
    q_learning_loss = 0

    for (state, action, reward, next_state) in batch:
        # Actor's loss
        q_require_grad = model.Q(state, action)

        with torch.no_grad():
            best_q = -float("inf")
            for other_action in env._actions:
                q_other_action = target_model.Q(next_state, other_action)

                if q_other_action > best_q:
                    best_q = q_other_action

        q_learning_loss += q_require_grad * (q_require_grad.detach() - (reward + model.env.gamma * best_q))

    return q_learning_loss


def get_actor_critic_loss(batch, model, target_model):
    actor_loss = 0
    critic_loss = 0

    for (state, action, reward, next_state) in batch:
        # Actor's loss
        q_require_grad = model.Q(state, action)
        v_require_grad = model.V(state)

        with torch.no_grad():
            v_target = target_model.V(next_state)
            entropy = model.pi_distribution(state).entropy()

        actor_loss += (q_require_grad - v_require_grad) * (
            q_require_grad.detach() - (reward + model.env.gamma * v_target)
        )

        # Critic loss
        critic_loss += v_require_grad * (
            v_require_grad.detach() - (reward + model.env.gamma * v_target + model.entropy_weight * entropy)
        )

    return actor_loss + critic_loss
