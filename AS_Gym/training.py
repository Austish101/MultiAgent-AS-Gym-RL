import csv
import time
from envs.AS_GymEnv import ASGymEnv
from envs.NoSim_GymEnv import GymEnv


def save_paths(x_axis, y_axis, z_axis, blue_count, red_count, states, obstacles, time_taken):
    with open("paths.csv", mode='w') as path_file:
        path_writer = csv.writer(path_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        path_writer.writerow([x_axis, y_axis, z_axis, blue_count, red_count, obstacles, time_taken])

        for i in range(0, len(states)):
            path_writer.writerow(states[i])


def read_agent_data(env, blue, red):
    env.Agent1.read_data("Agent1.csv")
    if blue >= 2:
        env.Agent2.read_data("Agent2.csv")
    if red >= 1:
        env.Agent3.read_data("Agent3.csv")
    if red >= 2:
        env.Agent4.read_data("Agent4.csv")


def get_actions(env, blue, red, observation):
    action2 = -1
    action3 = -1
    action4 = -1

    action1 = env.Agent1.get_action(observation[0])
    if blue >= 2:
        action2 = env.Agent2.get_action(observation[1])
    if red >= 1:
        action3 = env.Agent3.get_action(observation[2])
    if red >= 2:
        action4 = env.Agent4.get_action(observation[3])

    return [action1, action2, action3, action4]


def do_env_step(env, blue, red, actions):
    if (blue == 2) and (red == 2):
        observation, reward, done, info = env.step([actions[0], actions[1], actions[2], actions[3]])
    elif (blue == 2) and (red == 1):
        observation, reward, done, info = env.step([actions[0], actions[1], actions[2]])
    elif (blue == 2) and (red == 0):
        observation, reward, done, info = env.step([actions[0], actions[1]])
    elif (blue == 1) and (red == 0):
        observation, reward, done, info = env.step([actions[0]])
    elif (blue == 1) and (red == 1):
        observation, reward, done, info = env.step([actions[0], -1, actions[2]])
    elif (blue == 1) and (red == 2):
        observation, reward, done, info = env.step([actions[0], -1, actions[2], actions[3]])
    return observation, reward, done, info


def update_agents(env, blue, red, observation, actions, reward, info, done, steps, steps_in, total_steps, episodes):
    # if end of episode reached
    if (not done) and (steps == (steps_in - 1)):
        env.Agent1.update(info[0][0], info[0][1], actions[0], reward[0], observation[0], done)
        if blue >= 2:
            env.Agent2.update(info[1][0], info[1][1], actions[1], reward[1], observation[1], done)
        if red >= 1:
            env.Agent3.update(info[2][0], info[2][1], actions[2], 1, observation[2], done)
        if red >= 2:
            env.Agent3.update(info[3][0], info[3][1], actions[3], 1, observation[3], done)
        total_steps += steps
        if (episodes % 1000) == 0:
            print("Ep:", episodes, "destination not reached! Step:", steps)
    else:
        env.Agent1.update(info[0][0], info[0][1], actions[0], reward[0], observation[0], done)
        if blue >= 2:
            env.Agent2.update(info[1][0], info[1][1], actions[1], reward[1], observation[1], done)
        if red >= 1:
            env.Agent3.update(info[2][0], info[2][1], actions[2], reward[2], observation[2], done)
        if red >= 2:
            env.Agent3.update(info[3][0], info[3][1], actions[3], reward[3], observation[3], done)
    return total_steps


def save_agents(env, blue, red):
    env.Agent1.save_data('Agent1.csv')
    if blue >= 2:
        env.Agent2.save_data('Agent2.csv')
    if red >= 1:
        env.Agent3.save_data('Agent3.csv')
    if red >= 2:
        env.Agent4.save_data('Agent4.csv')


# main loop
def training_loop(
        is_airsim, reuse=False, blue=2, red=2, episodes_in=1000000, steps_in=100, rec_paths=False,
        obs_rate=0, rl_type=0, learning=True, moving_flag=False):
    # set environment
    if is_airsim:
        env = ASGymEnv(blue, red, rl_type, moving_flag)
    else:
        env = GymEnv(blue, red, obs_rate, rl_type, moving_flag)
        if rl_type == 1 or rl_type == 3:
            red = 0

    if reuse:
        # use saved data from previous saved run
        read_agent_data(env, blue, red)

    paths = []
    total_steps = 0
    start_time = time.time()

    # episode loop
    for episodes in range(0, episodes_in):

        # reset the environment
        observation = env.reset()

        # if paths will be saved
        if rec_paths:
            paths.append([[observation[0][1], observation[0][2]]])

        # step loop
        for steps in range(0, steps_in):
            # agent(s) calculate next action
            actions = get_actions(env, blue, red, observation)

            # movement/env step taken
            observation, reward, done, info = do_env_step(env, blue, red, actions)

            # update reinforcement learning
            if learning:
                total_steps = update_agents(
                    env, blue, red, observation, actions, reward, info, done, steps, steps_in, total_steps, episodes)

            # if saving paths
            if rec_paths:
                paths[episodes].append([observation[0][1], observation[0][2]])

            if done:
                total_steps += steps
                if (episodes % 1000) == 0:
                    print("Ep:", episodes, "destination reached! Step:", steps)
                break

    elapsed_time = time.time() - start_time
    print("Time elapsed: ", elapsed_time)
    print("Average steps over all episodes: ", total_steps / episodes_in)

    # save q table(s) at end of episodes
    save_agents(env, blue, red)

    # record data
    if rec_paths:
        if rl_type == 1 or rl_type == 3:
            obstacle_states = env.obstacle_states
        else:
            obstacle_states = env.obstacle_states[2:]
        save_paths(env.x_axis, env.y_axis, env.z_axis, blue, red, paths, obstacle_states, elapsed_time)

    env.close()
