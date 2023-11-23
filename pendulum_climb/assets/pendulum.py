import numpy as np
import pybullet as p
import os
import math


class Pendulum:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'pendulum.urdf')
        self.id = p.loadURDF(fileName=f_name,
                             basePosition=[0, 0, 1],
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
        # E.g. action_type 0 == add momentum, action_value = -20
        action_type, action_value = action

        # 0 = Apply Momentum
        # 1 = Grab
        # 2 = Release
        if action_type == 0:
            p.setJointMotorControl2(bodyIndex=self.id,
                                    jointIndex=self.joints[0],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=action_value)
        elif action_type == 1:
            pass

        elif action_type == 2:
            pass

    def get_observation(self):
        # Get the position and orientation of the pendulum in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.id, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))

        # Get the velocity of the pendulum
        vel = p.getBaseVelocity(self.id, self.client)[0][0:2]

        # Concatenate position, orientation, velocity
        # ([0.0, 0.0, 1.5], [1.0, 0.0, 0.0], 0.0)
        observation = (pos + ori + vel)

        return observation
