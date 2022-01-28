def compute_bad_steps(env, policy):
    bad_steps = 0

    for state in env._states:
        env.state = state
        next_state, _, _, _ = env.step(policy[state])
        if env.time_to_end[state] - env.time_to_end[next_state] != 1 and state != 5:
            bad_steps += 1

    return bad_steps
