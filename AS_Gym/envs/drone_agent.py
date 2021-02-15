from configparser import ConfigParser
import numpy as np
import math
from os.path import dirname, abspath, join
from airsim import MultirotorClient, ImageRequest, ImageType, DrivetrainType
import AS_GymEnv


class DroneAgent(MultirotorClient):
    def __init__(self):
        # connect to AirSim
        super().__init__()
        super().confirmConnection()
        super().enableApiControl(True)
        super().armDisarm(True)

        # get config data
        config = ConfigParser()
        config.read(join(dirname(dirname(abspath(__file__))), 'config.ini'))

        self.pos = self.state.kinematics_estimated.position
        self.target_pos = AS_GymEnv.destination
        self.state = self.getMultirotorState('drone1')
        self.qState = self.get_q_state()

        self.image_height = int(config['airsim_settings']['image_height'])
        self.image_width = int(config['airsim_settings']['image_width'])
        self.image_channels = int(config['airsim_settings']['image_channels'])
        self.image_size = self.image_height * self.image_width * self.image_channels

        self.action_mode = int(config['drone_agent']['action_mode'])
        # self.throttle = float(config['car_agent']['fixed_throttle'])
        # self.steering_granularity = int(config['car_agent']['steering_granularity'])
        # steering_max = float(config['car_agent']['steering_max'])
        # self.steering_values = np.arange(-steering_max, steering_max,
        #                                  2 * steering_max / (self.steering_granularity - 1)).tolist()
        # self.steering_values.append(steering_max)

        # self.start_pos = self.getPosition()

    def observe(self):
        # get RGB image from front camera, and mutlirotor state
        # spaces.Dict({
            #     'target': spaces.Box(low=-999, high=999, shape=3),
            #     'sensors': spaces.Dict({
            #         'position': spaces.Box(low=-999, high=999, shape=3),
            #         'velocity': spaces.Box(low=-1, high=1, shape=3),
            #         'camera': spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)

        size = 0
        while size != self.image_size:  # Sometimes simGetImages() return an unexpected resonpse.
            # If so, try it again.
            response = super().simGetImages([ImageRequest(0, ImageType.Scene, False, False)])[0]
            img1d_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            size = img1d_rgb.size

        img3d_rgb = img1d_rgb.reshape(self.image_height, self.image_width, self.image_channels)

        target = self.target_pos

        self.state = self.getMultirotorState('drone1')
        current_pos = self.state.kinematics_estimated.position
        current_vel = self.state.kinematics_estimated.linear_velocity

        return [target, [pos, vel, img3d_rgb]]

    def get_q_state(self):
        self.pos = self.state.kinematics_estimated.position
        x_grid = math.ceil((self.pos.x_val + 500) / 50)
        y_grid = math.ceil((self.pos.y_val + 500) / 50)
        z_grid = math.ceil((self.pos.z_val - 20) / 50)
        
        state = ...
        return state

    def restart(self):
        super.reset()
        super().enableApiControl(True)

    def move(self, movement, set_vel):
        self.state = self.getMultirotorState('drone1')
        self.pos = self.state.kinematics_estimated.position
        if movement == 0:
            self.moveToPositionAsync((self.pos.x_val + 50), self.pos.y_val, self.pos.z_val, set_vel, 10, DrivetrainType.ForwardOnly)
        elif movement == 1:
            self.moveToPositionAsync((self.pos.x_val - 50), self.pos.y_val, self.pos.z_val, set_vel, 10, DrivetrainType.ForwardOnly)
        elif movement == 2:
            self.moveToPositionAsync(self.pos.x_val, (self.pos.y_val + 50), self.pos.z_val, set_vel, 10, DrivetrainType.ForwardOnly)
        elif movement == 3:
            self.moveToPositionAsync(self.pos.x_val, (self.pos.y_val - 50), self.pos.z_val, set_vel, 10, DrivetrainType.ForwardOnly)
        elif movement == 4:
            self.moveToPositionAsync(self.pos.x_val, self.pos.y_val, (self.pos.z_val + 50), set_vel, 10, DrivetrainType.ForwardOnly)
        elif movement == 5:
            self.moveToPositionAsync(self.pos.x_val, self.pos.y_val, (self.pos.z_val - 50), set_vel, 10, DrivetrainType.ForwardOnly)
