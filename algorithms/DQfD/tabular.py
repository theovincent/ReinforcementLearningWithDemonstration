import numpy as np


def get_td_loss(transitions, env, Q, Q_target):
    return (
        transitions[0].reward
        + env.gamma * np.max([Q_target[transitions[0].next_state, action] for action in env._actions])
        - Q[transitions[0].state, transitions[0].action]
    )


def get_n_step_td_loss(transitions, env, Q, Q_target, gammas):
    return (
        np.sum([gammas[step] * transitions[step].reward for step in range(len(transitions))])
        + gammas[len(transitions)] * np.max([Q_target[transitions[-1].next_state, action] for action in env._actions])
        - Q[transitions[0].state, transitions[0].action]
    )


def get_expert_loss(transitions, env, Q, Q_target, diff_action_from_expert_penalisation):
    return (
        np.max(
            [
                Q_target[transitions[0].state, action]
                + diff_action_from_expert_penalisation * float(action != transitions[0].action)
                for action in env._actions
            ]
        )
        - Q[transitions[0].state, transitions[0].action]
    )


def tabular_DQfD(
    env,
    n_expert_trajectories,
    n_step_td,
    n_expert_iterations,
    n_rl_iterations,
    epsilon_decay_limit,
    update_target_frequency,
    td_loss_weight,
    n_td_loss_weight,
    expert_weight,
    diff_action_from_expert_penalisation,
    prioritized_buffer,
    weight_occurencies,
    show_policy=False,
    show_value_function=False,
    show_statistics=False,
    display_frequency=1,
):
    from algorithms.DQfD.replay_buffer import ReplayBuffer
    from algorithms.epsilon_greedy import EpsilonDecay

    gammas = [1]
    for pow_gamma in range(1, n_step_td + 1):
        gammas.append(env.gamma * gammas[-1])

    # Make the weights sum to one
    sum_weights = td_loss_weight + n_td_loss_weight + expert_weight
    td_loss_weight /= sum_weights
    n_td_loss_weight /= sum_weights
    expert_weight /= sum_weights

    if n_expert_trajectories > 0:
        expert_policy = env.expert_policy
    else:
        expert_policy = None

    replay_buffer = ReplayBuffer(
        env,
        EpsilonDecay(limit=epsilon_decay_limit),
        n_expert_trajectories,
        expert_policy,
        n_step_td,
        prioritized_buffer,
        weight_occurencies,
    )

    Q = np.ones((env.Ns, env.Na)) * 0
    Q_target = np.ones((env.Ns, env.Na)) * 0

    for expert_iteration in range(1, n_expert_iterations + 1):
        transitions, _, weight = replay_buffer.sample_n_transitions()

        if show_statistics and expert_iteration % display_frequency == 0:
            replay_buffer.display_statistics(transitions)

        td_loss = get_td_loss(transitions, env, Q, Q_target)
        n_step_td_loss = get_n_step_td_loss(transitions, env, Q, Q_target, gammas)
        expert_loss = get_expert_loss(transitions, env, Q, Q_target, diff_action_from_expert_penalisation)

        Q[transitions[0].state, transitions[0].action] += weight * (
            td_loss_weight * td_loss + n_td_loss_weight * n_step_td_loss + expert_weight * expert_loss
        )

        replay_buffer.td_losses[transitions[0].state][transitions[0].action] = td_loss

        if expert_iteration % update_target_frequency == 0:
            Q_target = Q.copy()

        if show_value_function and expert_iteration % display_frequency == 0:
            env.display_value_function(Q)
        if show_policy and expert_iteration % display_frequency == 0:
            env.display_policy(Q)

    print("End of expert phase")
    if show_value_function:
        env.display_value_function(Q)
    if show_policy:
        env.display_policy(Q)
    print("Beginning of rl phase")

    for rl_iteration in range(1, n_rl_iterations + 1):
        replay_buffer.collect_rl_transition(Q)

        transitions, is_expert, weight = replay_buffer.sample_n_transitions()
        weight = min(1, weight)  # keep convex combinaison

        # print(is_expert, weight)

        if show_statistics and rl_iteration % display_frequency == 0:
            replay_buffer.display_statistics(transitions)
            replay_buffer.display_rl_statistics()

        td_loss = get_td_loss(transitions, env, Q, Q_target)
        n_step_td_loss = get_n_step_td_loss(transitions, env, Q, Q_target, gammas)
        if is_expert:
            expert_loss = get_expert_loss(transitions, env, Q, Q_target, diff_action_from_expert_penalisation)
        else:
            expert_loss = 0

        Q[transitions[0].state, transitions[0].action] += weight * (
            td_loss_weight * td_loss + n_td_loss_weight * n_step_td_loss + expert_weight * expert_loss
        )

        replay_buffer.td_losses[transitions[0].state][transitions[0].action] = td_loss

        if rl_iteration % update_target_frequency == 0:
            Q_target = Q.copy()

        if show_value_function and rl_iteration % display_frequency == 0:
            env.display_value_function(Q)
        if show_policy and rl_iteration % display_frequency == 0:
            env.display_policy(Q)

    return Q, np.argmax(Q, axis=-1)
