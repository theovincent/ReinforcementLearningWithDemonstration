import numpy as np


def get_td_loss(transitions, env, Q, Q_target, gamma):
    return (
        transitions[0].reward
        + gamma * np.max([Q_target[transitions[0].next_state, action] for action in env._actions])
        - Q[transitions[0].state, transitions[0].action]
    )


def get_n_step_td_loss(transitions, env, Q, Q_target, gammas):
    return (
        np.sum([gammas[step] * transitions[step].reward for step in range(len(transitions) - 1)])
        + gammas[len(transitions)]
        * np.max([Q_target[transitions[len(transitions)].next_state, action] for action in env._actions])
        - Q[transitions[0].state, transitions[0].action]
    )


def get_expert_loss(transitions, env, Q, Q_target, diff_action_from_expert_penalisation):
    return (
        np.max(
            [
                Q_target[transitions[0].state, action]
                + diff_action_from_expert_penalisation * (action != transitions[0].action)
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
    epsilon_decay_limit=0.2,
    update_target_frequency=10,
    td_loss_weight=1,
    n_td_loss_weight=1,
    expert_weight=1,
    diff_action_from_expert_penalisation=0.8,
):
    from algorithms.DQfD.replay_buffer import ReplayBuffer
    from algorithms.epsilon_greedy import EpsilonDecay

    from simulators.grid_world import GAMMA

    gammas = [1]
    for pow_gamma in range(1, n_step_td):
        gammas.append(GAMMA * gammas[-1])

    if n_expert_trajectories > 0:
        from algorithms.VI_dynamic_programming import value_iteration

        _, expert_policy = value_iteration(env.P, env.R, GAMMA)
    else:
        expert_policy = None

    replay_buffer = ReplayBuffer(
        env,
        EpsilonDecay(limit=epsilon_decay_limit),
        n_expert_trajectories=n_expert_trajectories,
        expert_policy=expert_policy,
        n_step_td=n_step_td,
    )

    Q = np.zeros((env._states, env._actions))
    Q_target = np.zeros((env._states, env._actions))

    for expert_iteration in range(n_expert_iterations):
        transitions, _, weight = replay_buffer.sample_n_transitions()

        td_loss = get_td_loss(transitions, env, Q, Q_target, GAMMA)
        n_step_td_loss = get_n_step_td_loss(transitions, env, Q, Q_target, gammas)
        expert_loss = get_expert_loss(transitions, env, Q, Q_target, diff_action_from_expert_penalisation)

        Q[transitions[0].state, transitions[0].action] += weight * (
            td_loss_weight * td_loss + n_td_loss_weight * n_step_td_loss + expert_weight * expert_loss
        )

        replay_buffer.update_td_loss_last_sampled_transition(td_loss)

        if expert_iteration % update_target_frequency == 0:
            Q_target = Q.copy()

    for rl_iteration in range(n_rl_iterations):
        replay_buffer.collect_rl_transition(Q)

        transitions, is_expert, weight = replay_buffer.sample_n_transitions()

        td_loss = get_td_loss(transitions, env, Q, Q_target, GAMMA)
        n_step_td_loss = get_n_step_td_loss(transitions, env, Q, Q_target, gammas)
        if is_expert:
            expert_loss = get_expert_loss(transitions, env, Q, Q_target, diff_action_from_expert_penalisation)
        else:
            expert_loss = 0

        Q[transitions[0].state, transitions[0].action] += weight * (
            td_loss_weight * td_loss + n_td_loss_weight * n_step_td_loss + expert_weight * expert_loss
        )

        replay_buffer.update_td_loss_last_sampled_transition(td_loss)

        if rl_iteration % update_target_frequency == 0:
            Q_target = Q.copy()

    return Q
