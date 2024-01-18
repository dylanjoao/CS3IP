import numpy as np
import pybullet as p
import os
import math


class Torso:
    def __init__(self, client, pos):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'torso.xml')
        self.id = p.loadURDF(fileName=f_name,
                             basePosition=pos,
                             physicsClientId=client)

    def get_ids(self):
        return self.id, self.client

    def apply_action(self, action):
        pass

    def get_observation(self):
        pass
