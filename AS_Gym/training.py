import gym
import numpy as np
from envs.AS_GymEnv import AS_GymEnv
x_min = -500
x_max = 500
y_min = -500
y_max = 500
z_min = 20
z_max = 220
cube_size = 50
destinations = [[-475, -475, 45], [-475, 475, 45], [475, -475, 45], [475, 475, 45]]
actions = 6

env = AS_GymEnv(x_min, x_max, y_min, y_max, z_min, z_max, cube_size, len(destinations), actions)

# x_axis = (abs(x_min) + abs(x_max)) / cube_size
# y_axis = (abs(y_min) + abs(y_max)) / cube_size
# z_axis = (abs(z_min) + abs(z_max)) / cube_size
# obs_space = (x_axis * y_axis * z_axis) * total_destinations
# act_space = actions

x_axis = (abs(x_min) + abs(x_max)) / cube_size
y_axis = (abs(y_min) + abs(y_max)) / cube_size
z_axis = (abs(z_min) + abs(z_max)) / cube_size
locations = (x_axis * y_axis * z_axis)


def get_next_state(observation, action):
    # get next state given current state and action, prevent movement that isn't possible

    states_by_dests = states_in_env
    i = 1
    while observation > states_by_dests:
        i += 1
        states_by_dests = states_by_dests * i

    obs_location = observation - (states_by_dests - states_in_env)
    xy_plane = x_axis * y_axis

    # work out next state, if possible
    if action == 1:
        # if drone at max x, no move
        if obs_location <= x_axis:
            return observation
        for z in range(1, z_axis - 1):
            if (xy_plane * z) < obs_location <= ((xy_plane * z) + x_axis):
                return observation

        # next state:
        next_state = observation - x_axis

    elif action == 2:
        # if drone at min x, no move
        if (xy_plane - x_axis) < obs_location <= xy_plane:
            return observation
        for z in range(1, z_axis - 1):
            if ((xy_plane - x_axis) * z) < obs_location <= (xy_plane * z):
                return observation

        # next state:
        next_state = observation + x_axis

    elif action == 3:
        # if drone at max y, no move
        if obs_location % x_axis == 0:
            return observation

        # next state:
        next_state = observation + 1

    elif action == 4:
        # if drone at min y, no move
        if (obs_location - 1) % x_axis == 0:
            return observation

        # next state:
        next_state = observation - 1

    elif action == 5:
        # if drone at max z, no move
        if (xy_plane * (z_axis - 1)) < obs_location <= (xy_plane * z_axis):
            return observation

        # next state:
        next_state = observation + xy_plane

    elif action == 6:
        # if drone at min z, no move
        if obs_location <= xy_plane:
            return observation

        # next state:
        next_state = observation + xy_plane

    return next_state


def find_destination(destination):
    # given destination coords, find the state it resides in and update it
    target_coords = destinations[destination]
    obs_state = 0

# init Q-table:
# Each observation has the amount of actions in the action space, each action has (probability, nextstate, reward, done)
Q_table = np.zeros((env.observation_space.n + 1, env.action_space.n, 4))

# NOTE: obs index 0 is left empty for ease of use, 1 starts at min x, min y, min z (back, left, bottom
#   <x axis>  <z=1>       <z=2>
#   1, 2, 3,  <           10, 11, 12,
#   4, 5, 6,  y axis      13, 14, 15,     etc...
#   7, 8, 9,  >           16, 17, 18,

states_in_env = env.observation_space.n / len(destinations)

# fill Q-table with correct (probability, nextstate, reward, destination?)
for dests in range(0, len(destinations)):
    if dests == 0:
        for obs in range(1, states_in_env):
            for acs in range(0, env.action_space.n):
                Q_table[obs, acs, 0] = 1.0                          # probability
                Q_table[obs, acs, 1] = get_next_state(obs, acs)     # nextstate
                ...                                                 # rewards left empty
                Q_table[obs, acs, 3] = False                        # target destination?
    else:
        for obs in range(1, states_in_env):
            for acs in range(0, env.action_space.n):
                Q_table[obs, acs, 0] = 1.0                          # probability
                Q_table[obs, acs, 1] = Q_table[1, acs, 0]           # nextstate (same across destinations)
                ...                                                 # rewards left empty
                Q_table[obs, acs, 3] = False                       # target destination?
    find_destination(dests)


# Q = np.zeros(env.observation_space.n, env.action_space.n)
#
# eta = .628
# gma = .9
# epis = 5000
# rev_list = []
#
# for i in range(epis):
#     # reset
#     s = env.reset()
#     d = False
#
#     # learning
#     while d != True:
#         env.render()
#         # choose action
#         a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n) * (1./(i+1)))
#         # get new state and reward from env
