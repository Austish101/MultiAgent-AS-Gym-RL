# NOT USED

class DataClass:
    def __init__(self, drone1_pos, target_pos, drone1_cam):
        self.drone1_pos = drone1_pos
        self.target_pos = target_pos
        self.drone1_cam = drone1_cam

    def set_d1_pos(self, pos):
        self.drone1_pos = pos

    def set_target_pos(self, pos):
        self.target_pos = pos

    def set_d1_cam(self, cam):
        self.drone1_cam = cam
