import numpy as np
import pybullet as p
import os
import math


class Pendulum:
    def __init__(self, client, pos):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'pendulum.urdf')
        self.id = p.loadURDF(fileName=f_name,
                             basePosition=pos,
                             physicsClientId=client)

        # 2 Contact points
        self.constraints = [None, None]

    def apply_torque(self, joint, force):
        p.applyExternalTorque(objectUniqueId=self.id,
                              linkIndex=joint,
                              torqueObj=force,
                              flags=p.LINK_FRAME)

    def get_ids(self):
        return self.id, self.client

    def apply_action(self, action):

        # 0 = +VelX (1)
        # 1 = -VelX (1)
        # 2 = +VelY (1)
        # 3 = -VelY (1)
        # 4 = Grab (1)
        # 5 = Release (1)

        # 6 = +VelX (2)
        # 7 = -VelX (2)
        # 8 = +VelY (2)
        # 9 = -VelY (2)
        # 10 = Grab (2)
        # 11 = Release (2)

        joint = None
        if action < 6:
            joint = 0
        elif action > 5:
            joint = 1

        # +VelX
        if action == 0 or action == 6:
            self.apply_torque(joint, [150.0, 0, 0])

        # -VelX
        elif action == 1 or action == 7:
            self.apply_torque(joint, [-150.0, 0, 0])

        # +VelY
        elif action == 2 or action == 8:
            self.apply_torque(joint, [0.0, 150.0, 0])

        # -VelY
        elif action == 3 or action == 9:
            self.apply_torque(joint, [0.0, -150.0, 0])

        elif (action == 4 and self.constraints[0] is None) or (action == 10 and self.constraints[1] is None):
            in_range = self.target_in_range(joint)
            if in_range is not None:
                self.create_hold(joint, in_range.id)

        elif action == 3 or action == 7:
            self.remove_hold(joint)

        def create_hold(parent_link, child):
            pass

        def remove_hold(constraint):
            pass
