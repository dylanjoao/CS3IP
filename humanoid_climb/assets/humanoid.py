import random
import numpy as np
import pybullet as p
import os

from humanoid_climb.assets.robot_util import *


class Humanoid:

	def __init__(self, bullet_client, pos, ori, power, statefile=None, fixedBase=False):
		f_name = os.path.join(os.path.dirname(__file__), 'humanoid_symmetric.xml')

		self._p = bullet_client
		self.power = power

		self.robot = bullet_client.loadMJCF("mjcf/humanoid_symmetric.xml")[0]
		bullet_client.resetBasePositionAndOrientation(self.robot, pos, ori)
		if fixedBase:
			bullet_client.createConstraint(self.robot, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0, 1], pos)

		(self.parts, self.joints, self.ordered_joints, self.robot_body) \
			= addToScene(bullet_client, [self.robot])

		self.motor_names = ["abdomen_z", "abdomen_y", "abdomen_x"]
		self.motor_power = [100, 100, 100]
		self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
		self.motor_power += [100, 100, 300, 200]
		self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
		self.motor_power += [100, 100, 300, 200]
		self.motor_names += ["right_shoulder1", "right_shoulder2", "right_elbow"]
		self.motor_power += [75, 75, 75]
		self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
		self.motor_power += [75, 75, 75]
		self.motors = [self.joints[n] for n in self.motor_names]

	def apply_action(self, a):
		force_gain = 1
		for i, m, power in zip(range(17), self.motors, self.motor_power):
			m.set_motor_torque(float(force_gain * power * self.power * np.clip(a[i], -1, +1)))
