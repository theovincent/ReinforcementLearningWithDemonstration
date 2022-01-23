import os
from pathlib import Path

import numpy as np

from rlberry.envs import GridWorld


def get_time_to_end(env, state, policy):
    """The policy is expected to bring the agent to the terminal state."""
    env.state = state
    time_to_end = 0

    for timestep in range(500):
        action = policy[state]
        next_state, _, is_terminal, _ = env.step(action)
        state = next_state
        if is_terminal:
            break
        else:
            time_to_end += 1

    return time_to_end


def set_granular_reward(env, policy):
    reward_at = {}
    for coord, state in env.coord2index.items():
        if state == -1:
            continue
        reward_at[coord] = -get_time_to_end(env, state, policy)

    return Maze(env.grid_name, env.feature_type, env.dimensions, env.sigma, reward_at)


def set_normilized_reward(env):
    reward_at = env.reward_at
    rewards = [reward for reward in reward_at.values()]
    min_reward = min(rewards)
    max_reward = max(rewards)

    for coord in reward_at.keys():
        reward_at[coord] = (env.reward_at[coord] - min_reward) / (max_reward - min_reward)

    return Maze(env.grid_name, env.feature_type, env.dimensions, env.sigma, reward_at)


def divide_reward_by(env, factor):
    reward_at = env.reward_at
    for coord in env.reward_at.keys():
        reward_at[coord] = env.reward_at[coord] / factor

    return Maze(env.grid_name, env.feature_type, env.dimensions, env.sigma, reward_at)



class Maze(GridWorld):
    """Creates an instance of a simple grid-world MDP."""

    def __init__(self, grid_name, feature_type, dimensions=None, sigma=None, reward_at=None):
        from simulators.grid_world import GAMMA

        self.grid_name = grid_name
        self.feature_type = feature_type
        self.dimensions = dimensions
        self.sigma = sigma
        self.gamma = GAMMA

        if self.grid_name == "test":
            super().__init__(
                nrows=1,
                ncols=5,
                reward_at=reward_at
                if reward_at is not None
                else {(0, 0): -1.0, (0, 1): 0.0, (0, 2): 0.1, (0, 3): 0.5, (0, 4): 1.0},
                walls=None,
                success_probability=1,
                terminal_states=((0, 4),),
            )
        elif self.grid_name == "simple":
            super().__init__(
                nrows=5,
                ncols=7,
                reward_at=reward_at if reward_at is not None else {(0, 6): 1.0},
                walls=((0, 4), (1, 4), (2, 4), (3, 4)),
                success_probability=1,
                terminal_states=((0, 6),),
            )
        elif self.grid_name == "large":
            super().__init__(
                nrows=10,
                ncols=15,
                reward_at=reward_at if reward_at is not None else {(9, 14): 1.0},
                walls=(
                    # First wall
                    (0, 4),
                    (1, 4),
                    (2, 4),
                    (3, 4),
                    (4, 4),
                    (5, 4),
                    (6, 4),
                    (7, 4),
                    (8, 4),
                    # Second wall
                    (1, 9),
                    (2, 9),
                    (3, 9),
                    (4, 9),
                    (5, 9),
                    (6, 9),
                    (7, 9),
                    (8, 9),
                    (9, 9),
                ),
                success_probability=1,
                terminal_states=((9, 14),),
            )

        if self.feature_type != "one_hot":
            sim_matrix = np.zeros((self.Ns * self.Na, self.Ns))

            for state_i in range(self.Ns):
                for action_i in range(self.Na):
                    next_state_i, _, _, _ = self.sample(state_i, action_i)
                    row_next_state_i, col_next_state_i = self.index2coord[next_state_i]
                    prop_row_next_state_i = row_next_state_i / self.nrows
                    prop_col_next_state_i = col_next_state_i / self.ncols

                    for state_j in range(self.Ns):
                        row_state_j, col_state_j = self.index2coord[state_j]
                        prop_row_state_j = row_state_j / self.nrows
                        prop_col_state_j = col_state_j / self.ncols

                        dist = np.sqrt(
                            (prop_row_state_j - prop_row_next_state_i) ** 2.0
                            + (prop_col_state_j - prop_col_next_state_i) ** 2.0
                        )
                        sim_matrix[state_i * self.Na + action_i, state_j] = np.exp(-((dist / self.sigma) ** 2.0))

            _, _, vh = np.linalg.svd(sim_matrix.T)
            self.features = vh[: self.dimensions, :]
        else:
            self.features = None

    def get_feature(self, state, action):
        if self.feature_type == "one_hot":
            feature = np.zeros((self.S, self.A))
            feature[state, action] = 1

            return feature.flatten()
        else:
            return self.features[:, state * self.Na + action].copy()

    #
    # Code for rendering
    #
    def get_layout_array(self, state_data=None, fill_walls_with=np.nan):
        """
        Returns an array 'layout' of shape (nrows, ncols) such that:
            layout[row, col] = state_data[self.coord2idx[row, col]]
        If (row, col) is a wall:
            layout[row, col] = fill_walls_with
        Parameters
        ----------
        state_data : np.array, default = None
            Array of shape (self.observation_space.n,)
        fill_walls_with : float, default: np.nan
            Value to set in the layout in the coordinates corresponding to walls.
        Returns
        -------
        Gridworld layout array of shape (nrows, ncols).
        """
        layout = np.zeros((self.nrows, self.ncols))
        if state_data is not None:
            assert state_data.shape == (self.observation_space.n,)
            data_rows = [self.index2coord[idx][0] for idx in self.index2coord]
            data_cols = [self.index2coord[idx][1] for idx in self.index2coord]
            layout[data_rows, data_cols] = state_data
        else:
            state_rr, state_cc = self.index2coord[self.state]
            layout[state_rr, state_cc] = 1.0

        walls_rows = [ww[0] for ww in self.walls]
        walls_cols = [ww[1] for ww in self.walls]
        layout[walls_rows, walls_cols] = fill_walls_with
        return layout

    def get_layout_img(self, state_data=None, colormap_name="cool", wall_color=(0.0, 0.0, 0.0), min=None, max=None):
        """
        Returns an image array representing the value of `state_data` on
        the gridworld layout.
        Parameters
        ----------
        state_data : np.array, default = None
            Array of shape (self.observation_space.n,)
        colormap_name : str, default = 'cool'
            Colormap name.
            See https://matplotlib.org/tutorials/colors/colormaps.html
        wall_color : tuple
            RGB color for walls.
        Returns
        -------
        Gridworld image array of shape (nrows, ncols, 3).
        """
        import matplotlib
        from matplotlib import cm
        import matplotlib.pyplot as plt

        # map data to [0.0, 1.0]
        if state_data is not None:
            if min is None:
                min = state_data.min()
            if max is None:
                max = state_data.max()
            state_data = (state_data - min) / (max - min)

        colormap_fn = plt.get_cmap(colormap_name)
        layout = self.get_layout_array(state_data, fill_walls_with=np.nan)
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=colormap_fn)
        img = np.zeros(layout.shape + (3,))
        for rr in range(layout.shape[0]):
            for cc in range(layout.shape[1]):
                if np.isnan(layout[rr, cc]):
                    img[self.nrows - 1 - rr, cc, :] = wall_color
                else:
                    img[self.nrows - 1 - rr, cc, :3] = scalar_map.to_rgba(layout[rr, cc])[:3]
        return img

    def display_value_function(self, Q_or_V, from_value=False):
        import matplotlib.pyplot as plt

        if from_value:  # Q_or_V is V
            V = Q_or_V
        else:  # Q_or_V is Q
            V = Q_or_V.max(axis=1)

        plt.figure()
        plt.title("Value function")
        img = self.get_layout_img(V)
        plt.imshow(img)
        plt.show()

    def display_policy(self, Q_or_pi, from_pi=False):
        import matplotlib.pyplot as plt

        from simulators.grid_world import COMMANDS

        if from_pi:  # Q_or_pi is pi
            policy = Q_or_pi
        else:  # Q_or_pi is Q
            policy = np.argmax(Q_or_pi, axis=1)

        plt.figure()
        img = self.get_layout_img(policy, min=0, max=3)
        plt.title("Policy")
        plt.xlabel(COMMANDS)
        plt.imshow(img)
        plt.show()


def simulate_policy(policy, path_simulation, env, horizon):
    """Visualize a policy in an environment

    Args:
        policy: np.array
            matrix mapping states to action (Ns).
            If None, runs random policy.
        path_simulation: str
            path of the video to be store.
        env: GridWorld
            environment where to run the policy.
        horizon: int
            maximum number of timesteps in the environment.
    """
    from pyvirtualdisplay import Display

    reward_at = env.reward_at
    min_reward_at = np.min(list(reward_at.values()))
    for key in reward_at:
        reward_at[key] -= min_reward_at
    env_for_video = Maze(env.grid_name, env.feature_type, env.dimensions, env.sigma, reward_at)

    # To make the rendering possible
    display = Display(visible=0, size=(1400, 900))
    display.start()

    path_simulation = Path(path_simulation)
    if not path_simulation.parent.exists():
        os.mkdir(path_simulation.parent)

    env_for_video.enable_rendering()
    state = env_for_video.reset()  # get initial state
    for timestep in range(horizon):
        if policy is None:
            action = env_for_video.action_space.sample()  # take random actions
        else:
            action = policy[state]
        next_state, reward, is_terminal, info = env_for_video.step(action)
        state = next_state
        if is_terminal:
            break
    # save video and clear buffer
    env_for_video.save_video(str(path_simulation), framerate=5)
    env_for_video.clear_render_buffer()
    env_for_video.disable_rendering()
