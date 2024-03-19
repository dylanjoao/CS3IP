import pybullet as p
import os


class Target:
    def __init__(self, client, pos):
        f_name = os.path.join(os.path.dirname(__file__), 'target.urdf')
        self._p = client
        self.id = self._p.loadURDF(fileName=f_name, basePosition=pos)
        self.pos = pos
