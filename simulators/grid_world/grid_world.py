import numpy as np
import matplotlib.pyplot as plt

from rlberry.envs import GridWorld


def get_env():
    """Creates an instance of a grid-world MDP."""
    env = GridWorld(
        nrows=5,
        ncols=7,
        reward_at={(0, 6): 1.0},
        walls=((0, 4), (1, 4), (2, 4), (3, 4)),
        success_probability=1,
        terminal_states=((0, 6),),
    )
    return env


def simulate_policy(policy, name, env, horizon):
    """Visualize a policy in an environment

    Args:
        policy: np.array
            matrix mapping states to action (Ns).
            If None, runs random policy.
        name: str
            name of the video to be store.
        env: GridWorld
            environment where to run the policy.
        horizon: int
            maximum number of timesteps in the environment.
    """
    from pyvirtualdisplay import Display

    # To make the rendering possible
    display = Display(visible=0, size=(1400, 900))
    display.start()

    env.enable_rendering()
    state = env.reset()  # get initial state
    for timestep in range(horizon):
        if policy is None:
            action = env.action_space.sample()  # take random actions
        else:
            action = policy[state]
        next_state, reward, is_terminal, info = env.step(action)
        state = next_state
        if is_terminal:
            break
    # save video and clear buffer
    env.save_video(f"./videos/{name}.mp4", framerate=5)
    env.clear_render_buffer()
    env.disable_rendering()
