import numpy as np
import pybullet as p
import os
import math

# Reference https://www.gymlibrary.dev/environments/mujoco/humanoid/
class Torso:
    def __init__(self, client, pos):
        f_name = os.path.join(os.path.dirname(__file__), 'mjcf_torso.xml')

        self.client = client
        self.id = p.loadMJCF(mjcfFileName=f_name,
                             flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        self.motors = []
        self.motor_names = []
        self.motor_power = []
        self.ordered_joints = []
        self.ordered_joint_indices = []

        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/tensorflow/humanoid_running.py#L35
        self.human = self.id[0]
        self.ordered_joints = []
        self.ordered_joint_indices = []

        self.right_hand = -1
        self.left_hand = -1

        p.resetBasePositionAndOrientation(bodyUniqueId=self.id[0], posObj=pos, ornObj=[0.0, 0.0, 0.0, 1.0], physicsClientId=client)

        jdict = {}
        for j in range(p.getNumJoints(self.human)):
            info = p.getJointInfo(self.human, j)
            link_name = info[12].decode("ascii")
            if link_name == "left_foot": left_foot = j
            if link_name == "right_foot": right_foot = j
            if link_name == "left_wrist": self.left_hand = j
            if link_name == "right_wrist": self.right_hand = j
            self.ordered_joint_indices.append(j)

            if info[2] != p.JOINT_REVOLUTE: continue
            jname = info[1].decode("ascii")
            jdict[jname] = j
            lower, upper = (info[8], info[9])
            self.ordered_joints.append((j, lower, upper))

            p.setJointMotorControl2(self.human, j, controlMode=p.VELOCITY_CONTROL, force=0)

        self.motor_names = ["right_shoulder1", "right_shoulder2", "right_elbow"]
        self.motor_power = [75, 75, 75]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motors = [jdict[n] for n in self.motor_names]


    def get_ids(self):
        return self.id, self.client

    def apply_action(self, actions):
        forces = [0.] * len(self.motors)
        for m in range(len(self.motors)):
            limit = 15
            ac = np.clip(actions[m], -limit, limit)
            forces[m] = self.motor_power[m] * ac * 0.082
        p.setJointMotorControlArray(self.human, self.motors, controlMode=p.TORQUE_CONTROL, forces=forces)

    def attach(self, limb_link, target):
        pass

    def detach(self, limb):
        pass

    def get_observation(self):
        pass
