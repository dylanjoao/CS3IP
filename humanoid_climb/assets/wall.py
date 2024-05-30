import os
import pybullet as p


class Wall:
    def __init__(self, client, pos, ori, filename):
        f_name = os.path.join(os.path.dirname(__file__), 'wall.xml')

        self.client = client
        self.id = client.loadURDF(fileName=filename, basePosition=pos)
        client.resetBasePositionAndOrientation(self.id, pos, ori)
