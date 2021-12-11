import numpy as np
import matplotlib.pyplot as plt

from simulators.grid_world import COMMANDS


def display_value_function(env, w):
    plt.figure()
    Q = np.zeros((env.S, env.A))

    for state in env._states:
        for action in env._actions:
            Q[state, action] = env.get_feature(state, action) @ w

    V = Q.max(axis=1)

    print("Value function on terminal node:", V[env.coord2index[env.terminal_states[0]]])
    print("Value function on initial node:", V[0])
    plt.title("Value function")
    img = env.get_layout_img(V)
    plt.imshow(img)
    plt.show()


def display_policy(env, w):
    plt.figure()
    Q = np.zeros((env.S, env.A))

    for state in env._states:
        for action in env._actions:
            Q[state, action] = env.get_feature(state, action) @ w

    policy = np.argmax(Q, axis=1)

    img = env.get_layout_img(policy, min=0, max=3)
    plt.title("Policy")
    plt.xlabel(COMMANDS)
    plt.imshow(img)
    plt.show()
