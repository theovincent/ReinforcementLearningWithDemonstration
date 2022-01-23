import numpy as np


def get_Q(env, w):
    Q = np.zeros((env.S, env.A))

    for state in env._states:
        for action in env._actions:
            Q[state, action] = env.get_feature(state, action) @ w

    return Q


def lstd_grid_word(
    env,
    n_expert_samples,
    n_rl_samples,
    regularisor,
    regularisor_bellmann,
    max_iteration,
    epsilon_decay_limit,
    regularisor_expert,
    expert_loss_name,
    show_policy=False,
    show_value_function=False,
    show_statistics=False,
):
    from algorithms.API.replay_buffer import ReplayBuffer
    from algorithms.epsilon_greedy import EpsilonDecay
    from algorithms.API.optimizers import optimise_u, optimise_w
    from algorithms.API.losses import LossW

    if n_expert_samples > 0:
        from algorithms.VI_dynamic_programming import value_iteration

        _, expert_policy = value_iteration(env.P, env.R, env.gamma)
    else:
        expert_policy = None

    replay_buffer = ReplayBuffer(
        env, EpsilonDecay(limit=epsilon_decay_limit), n_expert_samples=n_expert_samples, expert_policy=expert_policy
    )

    loss_w = LossW(env, regularisor, regularisor_expert, expert_loss_name)

    w = np.zeros(env.dimensions)

    for iteration in range(max_iteration):
        # Exploration
        replay_buffer.collect_rl_samples(n_rl_samples, w, trajectory=False)
        if show_statistics:
            replay_buffer.display_statistics_on_samples()

        # Evaluation
        if regularisor_expert > 0:  # Add a constraint for the expert samples in the minimization
            samples_bellman = replay_buffer.buffer_rl
            samples_expert = replay_buffer.buffer_expert
        else:  # Add the expert samples as rl samples
            samples_bellman = replay_buffer.buffer_rl + replay_buffer.buffer_expert
            samples_expert = []

        u = optimise_u(env, samples_bellman, w, regularisor_bellmann)
        w = optimise_w(loss_w, w, samples_bellman, samples_expert, u, learning_rate=1)

        # Improvement
        # Done when samples are collected.
        if show_value_function:
            env.display_value_function(get_Q(env, w))
        if show_policy:
            env.display_policy(get_Q(env, w))

    Q = np.zeros((env.S, env.A))

    for state in env._states:
        for action in env._actions:
            Q[state, action] = env.get_feature(state, action) @ w

    return Q, np.argmax(Q, axis=1), replay_buffer
