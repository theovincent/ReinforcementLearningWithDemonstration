import numpy as np


def lstd_grid_word(
    env,
    n_expert_samples,
    n_rl_samples,
    regularisor,
    regularisor_bellmann,
    max_iteration=10,
    epsilon_decay_limit=0.2,
    regularisor_expert=False,
    show_policy=False,
    show_value_function=False,
    show_statistics=False,
):
    from algorithms.API.replay_buffer import ReplayBuffer
    from algorithms.epsilon_greedy import EpsilonDecay
    from algorithms.API.optimizers import optimise_u, optimise_w, optimise_w_with_demonstration

    from simulators.grid_world.displays import display_policy, display_value_function
    from simulators.grid_world import GAMMA

    if n_expert_samples > 0:
        from algorithms.VI_dynamic_programming import value_iteration

        _, expert_policy = value_iteration(env.P, env.R, GAMMA)
    else:
        expert_policy = None

    replay_buffer = ReplayBuffer(
        env, EpsilonDecay(limit=epsilon_decay_limit), n_expert_samples=n_expert_samples, expert_policy=expert_policy
    )
    size_feature = env.get_feature(0, 0).shape[0]

    w = np.zeros(size_feature)

    for iteration in range(max_iteration):
        # Exploration
        if iteration == 0:
            replay_buffer.collect_rl_samples(n_rl_samples - n_expert_samples, w, trajectory=False)
        else:
            replay_buffer.collect_rl_samples(n_rl_samples, w, trajectory=False)
        if show_statistics:
            replay_buffer.display_statistics_on_samples()

        # Evaluation
        if regularisor_expert > 0:  # Add a constraint of the expert samples in the minimization
            u = optimise_u(env, replay_buffer.buffer_rl, w, regularisor_bellmann, GAMMA)
            w = optimise_w_with_demonstration(
                env,
                replay_buffer.buffer_rl,
                u,
                regularisor,
                regularisor_expert=regularisor_expert,
                expert_samples=replay_buffer.buffer_expert,
            )
        else:  # Add the expert samples as rl samples
            u = optimise_u(env, replay_buffer.buffer_rl + replay_buffer.buffer_expert, w, regularisor_bellmann, GAMMA)
            w = optimise_w(env, replay_buffer.buffer_rl + replay_buffer.buffer_expert, u, regularisor)

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

    return Q, np.argmax(Q, axis=1), replay_buffer.buffer_rl
