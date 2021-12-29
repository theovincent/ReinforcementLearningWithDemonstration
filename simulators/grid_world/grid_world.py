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
        next_state, reward, is_terminal, info = env.step(action)
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
        reward_at[coord] = -get_time_to_end(env, state, policy) / len(env.coord2index)

    env.reward_at = reward_at


class Maze(GridWorld):
    """Creates an instance of a simple grid-world MDP."""

    def __init__(self, grid_name, feature_type, dimensions=None, sigma=None):
        from simulators.grid_world import GAMMA

        if grid_name == "test":
            super().__init__(
                nrows=1,
                ncols=5,
                reward_at={(0, 0): -1.0, (0, 1): 0.0, (0, 2): 0.1, (0, 3): 0.5, (0, 4): 1.0},
                walls=None,
                success_probability=1,
                terminal_states=((0, 4),),
            )
        elif grid_name == "simple":
            super().__init__(
                nrows=5,
                ncols=7,
                reward_at={(0, 6): 1.0},
                walls=((0, 4), (1, 4), (2, 4), (3, 4)),
                success_probability=1,
                terminal_states=((0, 6),),
            )
        elif grid_name == "large":
            super().__init__(
                nrows=10,
                ncols=15,
                reward_at={(9, 14): 1.0},
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

        self.gamma = GAMMA
        self.feature_type = feature_type

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
                        sim_matrix[state_i * self.Na + action_i, state_j] = np.exp(-((dist / sigma) ** 2.0))

            _, _, vh = np.linalg.svd(sim_matrix.T)
            self.features = vh[:dimensions, :]
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
            state_data = state_data - min
            if state_data.max() > 0.0:
                state_data = state_data / max

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

    def display_value_function(self, Q):
        import matplotlib.pyplot as plt

        V = Q.max(axis=1)

        plt.figure()
        plt.title("Value function")
        img = self.get_layout_img(V)
        plt.imshow(img)
        plt.show()

    def display_policy(self, Q):
        import matplotlib.pyplot as plt

        from simulators.grid_world import COMMANDS

        policy = np.argmax(Q, axis=1)

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

    # To make the rendering possible
    display = Display(visible=0, size=(1400, 900))
    display.start()

    path_simulation = Path(path_simulation)
    if not path_simulation.parent.exists():
        os.mkdir(path_simulation.parent)

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
    env.save_video(str(path_simulation), framerate=5)
    env.clear_render_buffer()
    env.disable_rendering()
