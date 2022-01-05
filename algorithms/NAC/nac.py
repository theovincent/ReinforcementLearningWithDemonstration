import numpy as np


def nac(env, n_expert_samples, n_expert_iterations, n_rl_iterations, update_target_frequecy):
    from algorithms.NAC.replay_buffer import ReplayBuffer
    from algorithms.NAC.model import SmallNAC

    replay_buffer = ReplayBuffer(n_expert_samples)
    model = SmallNAC(env)
    target_model = model.copy()

    for iteration in range(n_expert_iterations + n_rl_iterations):
        from_expert = True if iteration < n_expert_iterations else False

        if not from_expert:
            replay_buffer.collect_rl_sample(model)

        batch_expert = replay_buffer.get_batch(from_expert=from_expert)

        model.train_on_batch(batch_expert, target_model)

        if iteration % update_target_frequecy == 0:
            target_model = model.copy()

    return model
