from configparser import ConfigParser
from os.path import dirname, abspath, join
import math
import random
import numpy as np
import gym
import airsim
from gym import spaces, error, utils
from .drone_agent import DroneAgent
from q_learning import QAgent
from airsim import Vector3r


class ASGymEnv(gym.Env):
    """Custom Gym Environment to interface AirSim"""
    metadata = {'render.modes': ['human']}

    def __init__(self, blue_agents, red_agents, rl_type, moving_flag):
        super(ASGymEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        # setup values
        config = ConfigParser()
        config.read(join(dirname(dirname(abspath(__file__))), 'config.ini'))

        actions = int(config['drone_agent']['actions'])
        self.timeout = int(config['drone_agent']['timeout'])

        self.x_min = int(config['unreal_env']['x_min'])
        self.x_max = int(config['unreal_env']['x_max'])
        self.y_min = int(config['unreal_env']['y_min'])
        self.y_max = int(config['unreal_env']['y_max'])
        self.z_min = int(config['unreal_env']['z_min'])
        self.z_max = int(config['unreal_env']['z_max'])
        self.cube_size = int(config['unreal_env']['cube_size'])

        dest1 = [float(config['unreal_env']['destination_1_x']), float(config['unreal_env']['destination_1_y']),
                 float(config['unreal_env']['destination_1_z'])]
        dest2 = [float(config['unreal_env']['destination_2_x']), float(config['unreal_env']['destination_2_y']),
                 float(config['unreal_env']['destination_2_z'])]
        dest3 = [float(config['unreal_env']['destination_3_x']), float(config['unreal_env']['destination_3_y']),
                 float(config['unreal_env']['destination_3_z'])]
        dest4 = [float(config['unreal_env']['destination_4_x']), float(config['unreal_env']['destination_4_y']),
                 float(config['unreal_env']['destination_4_z'])]

        self.destinations = [dest1, dest2, dest3, dest4]

        self.x_axis = int((abs(self.x_min) + abs(self.x_max)) / self.cube_size)
        self.y_axis = int((abs(self.y_min) + abs(self.y_max)) / self.cube_size)
        self.z_axis = int((abs(self.z_min) + abs(self.z_max)) / self.cube_size)

        self.states_in_env = int(self.x_axis * self.y_axis * self.z_axis)
        coords0 = self.get_coords_of_state(0)
        coords0 = [coords0.x_val, coords0.y_val, coords0.z_val]
        coords_max = self.get_coords_of_state(self.states_in_env - 1)
        coords_max = [coords_max.x_val, coords_max.y_val, coords_max.z_val]
        self.max_distance = self.get_euclidean(coords0, coords_max, False)

        # ACTION SPACE
        # 6 movements, x+, x-, y+, y-, z+, z-
        self.action_space = spaces.Discrete(actions)

        # OBSERVATION SPACE
        # (amount possible locations) * the amount of destinations
        if blue_agents + red_agents == 4:
            self.observation_space = [spaces.Discrete(128), spaces.Discrete(128), spaces.Discrete(128),
                                      spaces.Discrete(128), spaces.Discrete(128)]
        elif blue_agents + red_agents == 3:
            self.observation_space = [spaces.Discrete(128), spaces.Discrete(128), spaces.Discrete(128),
                                      spaces.Discrete(128)]
        elif blue_agents + red_agents == 2:
            self.observation_space = [spaces.Discrete(128), spaces.Discrete(128), spaces.Discrete(128)]
        else:
            self.observation_space = [spaces.Discrete(128), spaces.Discrete(128)]
        self.obs_shape = 1 + blue_agents + red_agents

        # SETUP
        self.blue_agents = blue_agents
        self.red_agents = red_agents
        self.obs1_acs = 2
        self.obs2_acs = 2

        if (rl_type == 1) or (rl_type == 3):
            self.obstacle_drones = True
        else:
            self.obstacle_drones = False
        self.moving_flag = moving_flag
        if moving_flag:
            self.flag_acs = 2

        self.obstacle_states = [-1, -1]

        self.dest_state = -1
        self.drone_dest = DroneAgent("DroneDest")
        # setup drones
        if (rl_type == 0) or (rl_type == 1):
            # q-learning
            self.drone_agent1 = DroneAgent("Drone1")
            self.Agent1 = QAgent(False, self.states_in_env, self.observation_space, self.action_space, config)
            if blue_agents >= 2:
                self.drone_agent2 = DroneAgent("Drone2")
                self.Agent2 = QAgent(False, self.states_in_env, self.observation_space, self.action_space, config)
            if red_agents >= 1:
                self.drone_agent3 = DroneAgent("Drone3")
                self.Agent3 = QAgent(True, self.states_in_env, self.observation_space, self.action_space, config)
            if red_agents >= 2:
                self.drone_agent4 = DroneAgent("Drone4")
                self.Agent4 = QAgent(True, self.states_in_env, self.observation_space, self.action_space)
    
    def step(self, actions):
        # Execute one time step within the environment
        reward = [0, 0, 0, 0]

        if self.red_agents >= 1:
            self.Agent3, self.drone_agent3, reward[2], self.obs1_acs = self.move_counter_drone(
                self.Agent3, self.drone_agent3, actions[2], self.obs1_acs)
        if self.red_agents >= 2:
            self.Agent4, self.drone_agent4, reward[3], self.obs2_acs = self.move_counter_drone(
                self.Agent4, self.drone_agent4, actions[3], self.obs2_acs)

        # move path finding drones
        # agent1 always exists
        self.Agent1, self.drone_agent1, success1, blocked_by = self.move_drone(
            self.Agent1, self.drone_agent1, actions[0])
        if blocked_by != -1:
            reward[blocked_by] = 1
        if self.blue_agents >= 2:
            self.Agent2, self.drone_agent2, success2, blocked_by = self.move_drone(
                self.Agent2, self.drone_agent2, actions[1])
            if blocked_by != -1:
                reward[blocked_by] = 1

        # calculate rewards
        if self.blue_agents >= 2:
            reward, done = self.calculate_rewards(reward, success1, success2)
        else:
            reward, done = self.calculate_rewards(reward, success1)

        # moving destination? +y and -y
        if self.moving_flag:
            self.move_flag_lr()

        # return observations
        obs = self.format_observation()
        info = self.format_info(actions)

        # return the current env state, time taken to move, is episode done, info of states
        return obs, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        # get a random state and moves drone to corresponding position
        retry = True
        while retry:
            retry = False
            dest = random.randint(0, len(self.destinations) - 1)
            if not self.drone_dest.move(self.get_coords_of_state(dest)):
                retry = True
            self.Agent1.current_state = random.randint(0, self.states_in_env - 1)
            if not self.drone_agent1.move(self.get_coords_of_state(self.Agent1.current_state)):
                retry = True
            if self.blue_agents >= 2:
                self.Agent2.current_state = random.randint(0, self.states_in_env - 1)
                if not self.drone_agent2.move(self.get_coords_of_state(self.Agent2.current_state)):
                    retry = True
            if self.red_agents >= 1:
                self.Agent3.current_state = random.randint(0, self.states_in_env - 1)
                if not self.drone_agent3.move(self.get_coords_of_state(self.Agent3.current_state)):
                    retry = True
            if self.red_agents >= 2:
                self.Agent4.current_state = random.randint(0, self.states_in_env - 1)
                if not self.drone_agent4.move(self.get_coords_of_state(self.Agent4.current_state)):
                    retry = True

        self.dest_state = self.get_state_of_coords(self.destinations[dest])
        dest_pos = self.get_coords_of_state(self.dest_state)
        self.dest_coords = [dest_pos.x_val, dest_pos.y_val, dest_pos.z_val]
        obs = self.format_observation()

        return obs

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return  # not used but needed for Gym env

    def is_done(self, state):
        if self.dest_state == state:
            return True
        else:
            return False

    def move_counter_drone(self, agent, drone, action, obs_acs):
        reward = 0
        # if drones are an obstacle or learner
        if self.obstacle_drones:
            if self.get_next_state(agent.current_state, 2) == agent.current_state:
                new_state = self.get_next_state(agent.current_state, 3)
                new_pos = self.get_coords_of_state(new_state)
                if drone.move(new_pos):
                    agent.previous_state = agent.current_state
                    agent.current_state = self.get_next_state(agent.current_state, 3)
                    obs_acs = 3
                else:
                    obs_acs = 2
            elif self.get_next_state(agent.current_state, 3) == agent.current_state:
                new_state = self.get_next_state(agent.current_state, 2)
                new_pos = self.get_coords_of_state(new_state)
                if drone.move(new_pos):
                    agent.previous_state = agent.current_state
                    agent.current_state = self.get_next_state(agent.current_state, 2)
                    obs_acs = 2
                else:
                    obs_acs = 3
            else:
                new_state = self.get_next_state(agent.current_state, obs_acs)
                new_pos = self.get_coords_of_state(new_state)
                if drone.move(new_pos):
                    agent.previous_state = agent.current_state
                    agent.current_state = self.get_next_state(agent.current_state, obs_acs)
                else:
                    if obs_acs == 3:
                        obs_acs = 2
                    if obs_acs == 2:
                        obs_acs = 3
        else:
            new_state = self.get_next_state(agent.current_state, action)
            new_pos = self.get_coords_of_state(new_state)
            if drone.move(new_pos):
                reward = 0
                agent.previous_state = agent.current_state
                agent.current_state = new_state
            else:
                reward = -1
            self.obstacle_states[0] = agent.current_state
        return agent, drone, reward, obs_acs

    def move_drone(self, agent, drone, action):
        success = True
        blocked_by = -1
        new_state = self.get_next_state(agent.current_state, action)
        new_pos = self.get_coords_of_state(new_state)
        for i in range(0, len(self.obstacle_states)):
            # check if drone has been caught by a counter drone, or blocked in next move
            if (agent.current_state == self.obstacle_states[i]) or (new_state == self.obstacle_states[i]):
                # drone has been blocked, no move, unsuccessful
                success = False
                if i == 0:
                    blocked_by = 2
                elif i == 1:
                    blocked_by = 3
            elif (i == (len(self.obstacle_states) - 1)) and success:
                if drone.move(new_pos):
                    agent.previous_state = agent.current_state
                    agent.current_state = new_state
                else:
                    success = False
        return agent, drone, success, blocked_by

    def calculate_rewards(self, reward, success1, success2=False):
        done = False
        if success1:
            if self.is_done(self.Agent1.current_state):
                done = True
                reward[0] = 1
            else:
                reward[0] = 0
        else:
            reward[0] = -1
        if success2:
            if self.is_done(self.Agent2.current_state):
                done = True
                reward[1] = 1
                reward[0] = 1
            else:
                reward[1] = 0
        elif reward[0] == 1:
            reward[1] = 1
        else:
            reward[1] = -1

        # if reward is not the goal, or blocked, then get euclidean for team reward
        if (reward[0] == 0) or (reward[1] == 0):
            dest_pos = self.get_coords_of_state(self.dest_state)
            dest_coords = [dest_pos.x_val, dest_pos.y_val, dest_pos.z_val]

            pos1 = self.get_coords_of_state(self.Agent1.current_state)
            coords1 = [pos1.x_val, pos1.y_val, pos1.z_val]
            reward0 = self.get_euclidean([coords1[0], coords1[1], coords1[2]], dest_coords)
            pos2 = self.get_coords_of_state(self.Agent2.current_state)
            coords2 = [pos2.x_val, pos2.y_val, pos2.z_val]
            reward1 = self.get_euclidean([coords2[0], coords2[1], coords2[2]], dest_coords)
            if (reward[0] == 0) and (reward[1] == 0):
                if reward0 > reward1:
                    reward[0] = reward0
                    reward[1] = reward0
                    negative = - reward0
                else:
                    reward[0] = reward1
                    reward[1] = reward1
                    negative = - reward1
            elif reward[0] == 0:
                reward[0] = reward0
                negative = - reward0
            elif reward[1] == 0:
                reward[1] = reward1
                negative = - reward1
            # give red team negative reward if no other reward
            if reward[2] == 0:
                reward[2] = negative
            if reward[3] == 0:
                reward[3] = negative

        return reward, done

    def move_flag_lr(self):
        if self.get_next_state(self.dest_state, 2) == self.dest_state:
            new_dest = self.get_next_state(self.dest_state, 3)
            new_dest_pos = self.get_coords_of_state(new_dest)
            if self.drone_dest.move(new_dest_pos):
                self.dest_state = self.get_next_state(self.dest_state, 3)
                self.flag_acs = 3
            else:
                self.flag_acs = 2
        elif self.get_next_state(self.dest_state, 3) == self.dest_state:
            new_dest = self.get_next_state(self.dest_state, 2)
            new_dest_pos = self.get_coords_of_state(new_dest)
            if self.drone_dest.move(new_dest_pos):
                self.dest_state = self.get_next_state(self.dest_state, 2)
                self.flag_acs = 2
            else:
                self.flag_acs = 3
        else:
            new_dest = self.get_next_state(self.dest_state, self.flag_acs)
            new_dest_pos = self.get_coords_of_state(new_dest)
            if self.drone_dest.move(new_dest_pos):
                self.dest_state = self.get_next_state(self.dest_state, self.flag_acs)
            else:
                if self.flag_acs == 3:
                    self.flag_acs = 2
                if self.flag_acs == 2:
                    self.flag_acs = 3

    def get_euclidean(self, p, q, reward=True):
        # calculate the euclidean distance between 2 points
        distance = math.sqrt(((p[0] - q[0]) ** 2) + ((p[1] - q[1]) ** 2) + ((p[2] - q[2]) ** 2))
        if not reward:
            return distance

        # normalise between 0-1
        normalised = distance / self.max_distance
        # get the inverse, less distance = higher reward
        if normalised == 0:
            reward = normalised
        else:
            reward = (1/normalised) * 10
        return reward

    def format_observation(self):
        obs = [[-1, self.dest_state, [-1 for x in range(4)]] for y in range(4)]
        # set states
        obs[0][0] = self.Agent1.current_state
        obs[0][2][0] = self.Agent1.current_state
        obs[1][2][0] = self.Agent1.current_state
        obs[2][2][0] = self.Agent1.current_state
        obs[3][2][0] = self.Agent1.current_state
        if self.blue_agents >= 2:
            obs[1][0] = self.Agent2.current_state
            obs[0][2][1] = self.Agent2.current_state
            obs[1][2][1] = self.Agent2.current_state
            obs[2][2][1] = self.Agent2.current_state
            obs[3][2][1] = self.Agent2.current_state
        if self.red_agents >= 1:
            obs[2][0] = self.Agent3.current_state
            obs[0][2][2] = self.Agent3.current_state
            obs[1][2][2] = self.Agent3.current_state
            obs[2][2][2] = self.Agent3.current_state
            obs[3][2][2] = self.Agent3.current_state
        if self.red_agents >= 2:
            obs[3][0] = self.Agent4.current_state
            obs[0][2][3] = self.Agent4.current_state
            obs[1][2][3] = self.Agent4.current_state
            obs[2][2][3] = self.Agent4.current_state
            obs[3][2][3] = self.Agent4.current_state
        return obs

    def format_info(self, actions):
        info = [[-1 for x in range(2)] for y in range(4)]
        # set states
        info[0][0] = self.Agent1.previous_state
        info[0][1] = self.get_next_state(self.Agent1.previous_state, actions[0])
        if self.blue_agents >= 2:
            info[1][0] = self.Agent2.previous_state
            info[1][1] = self.get_next_state(self.Agent2.previous_state, actions[1])
        if self.red_agents >= 1:
            info[2][0] = self.Agent3.previous_state
            info[2][1] = self.get_next_state(self.Agent3.previous_state, actions[2])
        if self.red_agents >= 2:
            info[3][0] = self.Agent4.previous_state
            info[3][1] = self.get_next_state(self.Agent4.previous_state, actions[3])
        return info

    def get_coords_of_state(self, state):
        # get the location of the state on the 3d grid, unrelated to the destination
        xy_plane = self.x_axis * self.y_axis

        z = math.floor(state / xy_plane)
        # center of cube = min z + half of cube size + cube size times the amount of z cubes
        z_coord = self.z_min - (self.cube_size / 2) - (self.cube_size * z)

        x = math.floor((state - (xy_plane * z)) / self.x_axis)
        x_coord = self.x_min + (self.cube_size / 2) + (self.cube_size * x)

        y = (state - (xy_plane * z)) - (x * self.x_axis)
        y_coord = self.y_min + (self.cube_size / 2) + (self.cube_size * y)

        pos = Vector3r(x_coord, y_coord, z_coord)
        return pos

    def get_next_state(self, state, action):
        # get next state given current state and action, prevent movement that isn't possible
        xy_plane = self.x_axis * self.y_axis  # 16

        # work out next state, if not possible to move then action returns to same state
        if action == 0:
            # if drone at max x, no move
            # if (xy_plane - self.x_axis) <= state < xy_plane:
            #     return state
            for z in range(1, self.z_axis + 1):
                if ((xy_plane * z) - self.x_axis) <= state < (xy_plane * z):
                    return state
            # next state:
            next_state = state + self.x_axis

        elif action == 1:
            # if drone at min x, no move
            # if state < self.x_axis:
            #     return state
            for z in range(1, self.z_axis + 1):
                if state < self.x_axis:
                    return state
                if (xy_plane * z) <= state < ((xy_plane * z) + self.x_axis):
                    return state
            # next state:
            next_state = state - self.x_axis

        elif action == 2:
            # if drone at max y, no move
            if (state + 1) % self.x_axis == 0:
                return state
            # next state:
            next_state = state + 1

        elif action == 3:
            # if drone at min y, no move
            if state % self.x_axis == 0:
                return state
            # next state:
            next_state = state - 1

        elif action == 4:
            # if drone at max z, no move
            if (xy_plane * (self.z_axis - 1)) <= state:
                return state
            # next state:
            next_state = state + xy_plane

        elif action == 5:
            # if drone at min z, no move
            if state < xy_plane:
                return state
            # next state:
            next_state = state - xy_plane
        else:
            next_state = state

        if next_state < 0:
            return state
        return next_state


    def get_state_of_coords(self, coords):
        # given coords, find the state they reside in

        # get which x, y, and z cubes the target resides in
        x_count = self.x_min + self.cube_size
        x = 0
        while coords[0] >= x_count:
            x += 1
            x_count += self.cube_size
        y_count = self.y_min + self.cube_size
        y = 0
        while coords[1] >= y_count:
            y += 1
            y_count += self.cube_size
        z_count = self.z_min + self.cube_size
        z = 0
        while coords[2] >= z_count:
            z += 1
            z_count -= self.cube_size

        xy_plane = self.x_axis * self.y_axis
        obs_state = 0

        for i in range(0, x):
            obs_state = obs_state + self.x_axis
        for i in range(0, y):
            obs_state = obs_state + 1
        for i in range(0, z):
            obs_state = obs_state + xy_plane

        return obs_state

    def close(self):
        self.drone_agent1.reset()
        # restarts the client, only needs to be done once for any drone
