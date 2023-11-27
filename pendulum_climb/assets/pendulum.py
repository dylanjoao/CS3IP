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

        # Joint indices as found by p.getJointInfo()
        self.joints = [0, 1]

        # Joint speed
        self.top_momentum = 0

        # Target grasped
        self.top_held = None

        self.targets = []

    def get_ids(self):
        return self.id, self.client

    def apply_action(self, action):

        # 0 = +Vel
        # 1 = -Vel
        # 2 = Grab
        # 3 = Release
        if action == 0:
            p.setJointMotorControl2(bodyIndex=self.id,
                                    jointIndex=self.joints[0],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=100.0)

        elif action == 1:
            p.setJointMotorControl2(bodyIndex=self.id,
                                    jointIndex=self.joints[0],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=-100.0)

        elif action == 2 and self.top_held is None:
            link_state = p.getLinkState(self.id, 0)
            target_in_range = None

            for i in range(len(self.targets)):
                target_pos, _ = p.getBasePositionAndOrientation(self.targets[i].id)
                distance = np.linalg.norm(np.array(link_state[0]) - np.array(target_pos))
                if distance < 0.1:
                    target_in_range = self.targets[i]
                    break

            if target_in_range is not None:
                self.create_hold(target_in_range.id)

        elif action == 3:
            self.remove_hold()

    def remove_hold(self):
        if self.top_held is not None:
            p.removeConstraint(self.top_held)
            self.top_held = None

    def create_hold(self, target):
        self.remove_hold()
        constraint = p.createConstraint(parentBodyUniqueId=self.id,
                                        parentLinkIndex=0,
                                        childBodyUniqueId=target,
                                        childLinkIndex=-1,
                                        jointType=p.JOINT_POINT2POINT,
                                        jointAxis=[0, 0, 0],
                                        parentFramePosition=[0, 0, 0],
                                        childFramePosition=[0, 0, 0])

        self.top_held = constraint

    def get_observation(self):
        # Get the position and orientation of the pendulum in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.id, self.client)
        ang = p.getEulerFromQuaternion(ang)

        # Get the velocity of the pendulum
        vel = p.getBaseVelocity(self.id, self.client)[0]

        # Concatenate position, orientation, velocity
        # ([0.0, 0.0, 1.5], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        observation = (pos + ang + vel)


        return observation
