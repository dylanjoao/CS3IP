from typing import List

import pybullet as p
import os


class Target:
    def __init__(self, client, pos, ori: List = [0, 0, 0, 1], path: str = "E:\\Programs\\GymRL\\PyBullet\\CS3IP\\CS3IP\\humanoid_climb\\assets\\target.xml"):
        f_name = os.path.join(os.path.dirname(__file__), 'target.xml')
        self.pos = pos
        self.id = client.loadURDF(fileName=path, basePosition=pos)
        client.resetBasePositionAndOrientation(self.id, pos, ori)
        client.setCollisionFilterGroupMask(self.id, -1, 0, 0)