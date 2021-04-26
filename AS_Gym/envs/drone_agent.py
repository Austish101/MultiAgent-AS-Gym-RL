from configparser import ConfigParser
from os.path import dirname, abspath, join

import airsim
from airsim import MultirotorClient, ImageRequest, ImageType, DrivetrainType, Vector3r


class DroneAgent():
    def __init__(self, vehicle_name):
        self.name = vehicle_name
        self.client = airsim.MultirotorClient()

        # connect to AirSim
        # MultirotorClient.__init__(self)
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.name)
        self.client.armDisarm(True, self.name)
        self.client.takeoffAsync(vehicle_name=self.name)
        # MultirotorClient.hoverAsync(self, vehicle_name=self.name)
        self.pos = self.get_pos()
        # print("Pos:", self.pos.x_val, self.pos.y_val, self.pos.z_val)

        config = ConfigParser()
        config.read(join(dirname(dirname(abspath(__file__))), 'config.ini'))
        self.set_vel = int(config['drone_agent']['velocity'])
        self.timeout = int(config['drone_agent']['timeout'])

    def observe(self):
        return self.get_pos()

    def restart(self):
        self.client.reset()
        # super.reset()
        self.client.enableApiControl(True, self.name)

    def move(self, new_pos):
        self.client.enableApiControl(True, self.name)
        self.client.armDisarm(True, self.name)

        start_time = self.client.getMultirotorState(vehicle_name=self.name).timestamp
        self.pos = self.get_pos()

        print(self.name)
        print("Move from", self.pos.x_val, self.pos.y_val, self.pos.z_val)
        print("Move to", new_pos.x_val, new_pos.y_val, new_pos.z_val)

        self.client.moveToPositionAsync(new_pos.x_val, new_pos.y_val, new_pos.z_val, self.set_vel,
                                 timeout_sec=self.timeout, vehicle_name=self.name).join()
        while self.get_pos().distance_to(new_pos) > 1:
            if self.client.simGetCollisionInfo(vehicle_name=self.name).has_collided:
                if self.client.simGetCollisionInfo(vehicle_name=self.name).time_stamp > start_time:
                    # if collision, return to last position
                    self.client.moveToPositionAsync(self.pos.x_val, self.pos.y_val, self.pos.z_val, self.set_vel,
                                             timeout_sec=self.timeout, vehicle_name=self.name).join()
                    if self.get_pos().distance_to(self.pos) < 1:
                        return False
            else:
                self.client.moveToPositionAsync(new_pos.x_val, new_pos.y_val, new_pos.z_val, self.set_vel,
                                         timeout_sec=self.timeout, vehicle_name=self.name).join()

        self.pos = self.get_pos()
        return True

    def get_pos(self):
        state = self.client.getMultirotorState(vehicle_name=self.name)
        return state.kinematics_estimated.position
