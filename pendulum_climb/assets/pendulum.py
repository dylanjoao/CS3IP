import numpy as np
import pybullet as p
import os
import math


class Pendulum:
    def __init__(self, client, pos):
        self._p = client
        f_name = os.path.join(os.path.dirname(__file__), 'pendulum.urdf')
        self.id = self._p.loadURDF(fileName=f_name, basePosition=pos)

        self.inital_state = self._p.getBasePositionAndOrientation(self.id)

        # Joint indices as found by p.getJointInfo()
        self.joints = [0, 1]
        self.constraints = [None, None]

        # Joint speed
        self.top_momentum = 0

        # Target grasped
        self.top_held = None
        self.bottom_hold = None

        self.targets = []

    def reset_state(self):
        self.remove_hold(0)
        self.remove_hold(1)
        self._p.resetBasePositionAndOrientation(self.id, self.inital_state[0], self.inital_state[1])

    def get_ids(self):
        return self.id

    def apply_action(self, action):

        # 0 = +Vel (1)
        # 1 = -Vel (1)
        # 2 = Grab (1)
        # 3 = Release (1)

        # 4 = +Vel (2)
        # 5 = -Vel (2)
        # 6 = Grab (2)
        # 7 = Release (2)

        joint = None
        if action < 4:
            joint = 0
        elif action > 3:
            joint = 1

        if action == 0 or action == 4:
            self._p.setJointMotorControl2(bodyIndex=self.id,
                                          jointIndex=joint,
                                          controlMode=p.VELOCITY_CONTROL,
                                          targetVelocity=150.0)

        elif action == 1 or action == 5:
            self._p.setJointMotorControl2(bodyIndex=self.id,
                                          jointIndex=joint,
                                          controlMode=p.VELOCITY_CONTROL,
                                          targetVelocity=-150.0)

        elif (action == 2 and self.top_held is None) or (action == 6 and self.bottom_hold is None):
            in_range = self.target_in_range(joint)
            if in_range is not None:
                self.create_hold(joint, in_range.id)

        elif action == 3 or action == 7:
            self.remove_hold(joint)

    def target_in_range(self, joint):
        link_state = self._p.getLinkState(self.id, joint)
        target_in_range = None

        for i in range(len(self.targets)):
            target_pos, _ = self._p.getBasePositionAndOrientation(self.targets[i].id)
            distance = np.linalg.norm(np.array(link_state[0]) - np.array(target_pos))
            if distance < 0.1:
                target_in_range = self.targets[i]
                break

        return target_in_range

    def remove_hold(self, joint):

        if joint is 0 and self.top_held is not None:
            self._p.removeConstraint(self.top_held)
            self.top_held = None

        if joint is 1 and self.bottom_hold is not None:
            self._p.removeConstraint(self.bottom_hold)
            self.bottom_hold = None

    def create_hold(self, joint, target):
        self.remove_hold(joint)
        constraint = self._p.createConstraint(parentBodyUniqueId=self.id,
                                              parentLinkIndex=joint,
                                              childBodyUniqueId=target,
                                              childLinkIndex=-1,
                                              jointType=p.JOINT_POINT2POINT,
                                              jointAxis=[0, 0, 0],
                                              parentFramePosition=[0, 0, 0],
                                              childFramePosition=[0, 0, 0])

        if joint is 0:
            self.top_held = constraint
        if joint is 1:
            self.bottom_hold = constraint

    def get_observation(self):
        ###
        #   Agent centre position   (x y z)
        #   Agent velocity          (x y z)
        #   Agent angle             (x y z)
        #   Joint Held              (1, 1)

        # Get the position and orientation of the pendulum in the simulation
        pos, ang = self._p.getBasePositionAndOrientation(self.id)
        ang = self._p.getEulerFromQuaternion(ang)
        link_state = self._p.getLinkStates(self.id, self.joints)
        eff1 = link_state[0][0]
        eff2 = link_state[1][0]

        # Get the velocity of the pendulum
        vel = self._p.getBaseVelocity(self.id)[0]

        holds = [0 if self.top_held is None else 1] + [0 if self.bottom_hold is None else 1]

        # Concatenate holds[2], position[3], angle[3], velocity[3]
        # ([1, 0], [0.0, 0.0, 1.5], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        observation = np.concatenate((holds, pos, ang, vel), dtype=np.float32)

        return holds, pos, ang, vel, eff1, eff2
