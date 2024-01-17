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
        self.targets = []

    def apply_torque(self, joint, force):
        p.applyExternalTorque(objectUniqueId=self.id,
                              linkIndex=joint,
                              torqueObj=force,
                              flags=p.LINK_FRAME,
                              physicsClientId=self.client)

    def get_ids(self):
        return self.id, self.client

    def apply_action(self, action):

        # 0 = +VelX (1)
        # 1 = -VelX (1)
        # 2 = Grab (1)
        # 3 = Release (1)

        # 4 = +VelX (2)
        # 5 = -VelX (2)
        # 6 = Grab (2)
        # 7 = Release (2)

        joint = None
        if action < 4:
            joint = 0
        elif action > 3:
            joint = 1

        # +VelX
        if action == 0 or action == 4:
            self.apply_torque(joint, [150.0, 0, 0])

        # -VelX
        elif action == 1 or action == 5:
            self.apply_torque(joint, [-150.0, 0, 0])

        elif (action == 2 and self.constraints[0] is None) or (action == 6 and self.constraints[1] is None):
            in_range = self.target_in_range(joint)
            if in_range is not None:
                self.create_hold(joint, in_range.id)

        elif action == 3 or action == 7:
            self.remove_hold(joint)

    def create_hold(self, parent_link, child):
        self.remove_hold(parent_link)
        constraint = p.createConstraint(parentBodyUniqueId=self.id,
                                        parentLinkIndex=parent_link,
                                        childBodyUniqueId=child,
                                        childLinkIndex=-1,
                                        jointType=p.JOINT_POINT2POINT,
                                        jointAxis=[0, 0, 0],
                                        parentFramePosition=[0, 0, 0],
                                        childFramePosition=[0, 0, 0],
                                        physicsClientId=self.client)

        self.constraints[parent_link] = constraint

    def remove_hold(self, link):
        if self.constraints[link] is None:
            return

        p.removeConstraint(self.constraints[link], physicsClientId=self.client)
        self.constraints[link] = None

    def target_in_range(self, link):
        link_state = p.getLinkState(self.id, link, physicsClientId=self.client)
        target_in_range = None

        for i in range(len(self.targets)):
            target_pos, _ = p.getBasePositionAndOrientation(self.targets[i].id, physicsClientId=self.client)
            distance = np.linalg.norm(np.array(link_state[0]) - np.array(target_pos))
            if distance < 0.2:
                target_in_range = self.targets[i]
                break

        return target_in_range
