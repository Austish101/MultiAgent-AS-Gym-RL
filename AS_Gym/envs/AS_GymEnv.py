from configparser import ConfigParser
from math import sqrt
import random

import numpy as np
import gym
import airsim
from gym import spaces, error, utils
from os.path import dirname, abspath, join
from .drone_agent import DroneAgent


class AS_GymEnv(gym.Env):
    """Custom Gym Environment to interface AirSim"""
    metadata = {'render.modes': ['human']}

    def __init__(self, x_max, x_min, y_max, y_min, z_max, z_min, cube_size, dests, actions):
        super(AS_GymEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        config = ConfigParser()
        config.read(join(dirname(dirname(abspath(__file__))), 'config.ini'))

        # ACTION SPACE
        # moveByRollPitchYawThrottle
        # Actions: roll,pitch,yaw,throttle \lowest accepted values/   \highest accepted values/
        # self.action_space = spaces.Box(np.array([-999, -999, 0, 0]), np.array([+999, +999, +999, 100]))
        # moveToPosition
        # Actions: x, y, z, velocity
        # 6 movements, x+, x-, y+, y-, z+, z-
        self.action_space = spaces.Discrete(actions)

        # OBSERVATION SPACE
        # Get image config:
        # self.image_height = int(config['airsim_settings']['image_height'])
        # self.image_width = int(config['airsim_settings']['image_width'])
        # self.image_channels = int(config['airsim_settings']['image_channels'])
        # image_shape = (self.image_height, self.image_width, self.image_channels)

        # # Using image as input:
        # self.observation_space = spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)

        # getMultirotorState

        # (amount possible locations) * the amount of destinations
        obs_space = (x_axis * y_axis * z_axis) * len(dests)
        self.observation_space = spaces.Discrete(obs_space)

        self.destination = dests[random.randint(0, len(dests) - 1)]

        # self.nested_observation_space = spaces.Dict({
        #     'target': spaces.Box(low=min, high=max, shape=3),
        #     'sensors': spaces.Dict({
        #         'position': spaces.Box(low=min, high=max, shape=3),
        #         'velocity': spaces.Box(low=-1, high=1, shape=3)
        #         'camera': spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
        #     })
        # })

        self.drone_agent = DroneAgent()

    def step(self, movement):
        # Execute one time step within the environment

        # Execute a movement according to the action
        self.drone_agent.move(movement)

        # Compute Reward
        # MultirotorState: Returns collision, kine, and timestamp -
        #     Kine returns position, orientation, linear/angular velocity/acceleration
        drone_state = self.drone_agent.getMultirotorState()
        reward = self.calculate_reward(drone_state)
        # reward = self.calculate_reward(action)

        # Has the target been reached?
        done = self.is_done(reward, drone_state)
        # done = False
        # self.epoch = self.epoch + 1
        # if self.epoch == len(self.data):
        #   done = True

        # Log info - ("x_pos": x_val, "y_pos": y_val)
        # info = {}

        # Get Observation
        observation = self.drone_agent.observe()
        # observation = {**self.notifications.iloc[self.epoch].to_dict(), **self.contexts.iloc[self.epoch].to_dict()}

        return observation, reward, done  # , info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.drone_agent.restart()
        observation = self.drone_agent.observe()
        return observation

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return  # not used but needed for Gym

    def is_done(self, reward, state, start_time):
        # if reward < 0 (drone at target)
        # OR if time elapsed == 1 min
        #   return True
        # else return False
        
        if reward < 0:
            return True
        # elif (start_time - state.timestamp) > 1:
        #     return True
        else:
            return False


    def calculate_reward(self, state, target, start_pos):
        # calculate current distance from target using kine
        x = state.kinematics_estimated.position[0]
        y = state.kinematics_estimated.position[1]
        z = state.kinematics_estimated.position[2]
        tx = target[0]
        ty = target[1]
        tz = target[2]
        # d = sqrt(x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2
        distance = sqrt(((tx - x) * (tx - x)) + ((ty - y) * (ty - y)) + ((tz - z) * (tz - z)))

        # get max distance
        sx = start_pos[0]
        sy = start_pos[1]
        sz = start_pos[2]
        start_distance = sqrt(((tx - sx) * (tx - sx)) + ((ty - sy) * (ty - sy)) + ((tz - sz) * (tz - sz)))

        # calculate reward as between 0-1, 1 for on target
        # if drone has moved further away from target than its starting place, reward = 0
        if distance > start_distance:
            return 0
        # else normalise between 0 and 1, where 1 is within 1 meter of target
        else:
            reward = distance / (start_distance - 1)
        return  # reward

    def close(self):
        self.drone_agent.reset()
        return
