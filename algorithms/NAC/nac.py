from functools import partial
import numpy as np

import torch


def nac(
    env,
    loss_name,
    entropy_weight,
    n_expert_samples,
    n_expert_iterations,
    n_rl_iterations,
    batch_size,
    update_target_frequency,
    show_policy=False,
    show_value_function=False,
    show_statistics=False,
    display_frequency=1,
):
    from algorithms.NAC.replay_buffer import ReplayBuffer
    from algorithms.NAC.model import SmallNAC
    from algorithms.NAC.losses import get_q_learning_loss, get_actor_critic_loss

    if n_expert_samples > 0:
        from algorithms.VI_dynamic_programming import value_iteration

        _, expert_policy = value_iteration(env.P, env.R, env.gamma)
    else:
        expert_policy = None

    replay_buffer = ReplayBuffer(
        env,
        n_expert_samples,
        expert_policy,
        batch_size,
    )
    model = SmallNAC(env, entropy_weight)
    target_model = SmallNAC(env, entropy_weight)
    model.copy_to(target_model)

    if loss_name == "q_learning":
        get_loss = partial(get_q_learning_loss, env)
    elif loss_name == "actor_critic":
        get_loss = get_actor_critic_loss

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for iteration in range(1, n_expert_iterations + n_rl_iterations + 1):
        from_expert = True if iteration <= n_expert_iterations else False

        if iteration > n_expert_iterations:
            optimizer.param_groups[0]["lr"] = 1e-4

        if not from_expert:
            replay_buffer.collect_rl_sample(model)

        batch = replay_buffer.get_batch(from_expert=from_expert)

        if show_statistics and iteration % display_frequency == 0:
            replay_buffer.display_statistics(from_expert)

        loss = get_loss(batch, model, target_model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if show_value_function and iteration % display_frequency == 0:
            value_function = np.zeros((env.Ns))

            for state in env._states:
                value_function[state] = model.V(state)
            env.display_value_function(value_function, from_value=True)
        if (show_policy and iteration % display_frequency == 0) or iteration == n_expert_iterations + 1:
            if iteration == n_expert_iterations + 1:
                print("End expert iterations")
            policy = np.zeros((env.Ns))

            for state in env._states:
                policy[state] = np.argmax(model.pi_distribution(state).probs.detach().numpy())
            env.display_policy(policy, from_pi=True)

        if iteration % update_target_frequency == 0:
            model.copy_to(target_model)

    policy = np.zeros((env.Ns), dtype=int)

    for state in env._states:
        policy[state] = np.argmax(model.pi_distribution(state).probs.detach().numpy())

    return model, policy
