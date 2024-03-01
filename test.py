import math
import time
from time import sleep
import pybullet as p
import pybullet_data
import numpy as np
import os

from pendulum_climb.assets.pendulum import Pendulum

current_directory = os.getcwd()

# Can alternatively pass in p.DIRECT
client = p.connect(p.GUI)
p.setGravity(0, 0, -9.8, physicsClientId=client)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(1)


def apply_action(body, actions, motors, power):
	forces = [0. for i in range(len(motors))]
	for m in range(len(motors)):
		limit = 15
		ac = np.clip(actions[m], -limit, limit)
		forces[m] = power[m] * ac
	p.setJointMotorControlArray(body, motors, controlMode=p.TORQUE_CONTROL, forces=forces)


# flags = p.URDF_MAINTAIN_LINK_ORDER + p.URDF_USE_SELF_COLLISION
flags = p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
plane = p.loadURDF("plane.urdf")
# humanoid = p.loadURDF(current_directory + "/torso_climb/assets/pyb_torso_2.xml", basePosition=[0, 0, 2], baseOrientation=[0.707, 0, 0, 0.707], flags=flags, globalScaling=0.25, useFixedBase=True)
# humanoid = p.loadURDF(current_directory + "/torso_climb/assets/pyb_torso.xml", basePosition=[0, 0, 2], flags=flags, useFixedBase=True)
humanoid = p.loadMJCF(current_directory + "/torso_climb/assets/mjcf_torso.xml")
p.createConstraint(humanoid[0], -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0, ], [0, 0, 1])

debug_params = []
motors = []

body = humanoid[0]
for i in range(p.getNumJoints(body)):
	ji = p.getJointInfo(body, i)
	if ji[2] == p.JOINT_REVOLUTE:
		motors.append(i)
		debug_params.append(p.addUserDebugParameter(ji[1].decode("utf-8"), -1, 1, 0.0))

power = [75 for i in range(len(debug_params))]

while True:
	# p.stepSimulation()
	events = p.getKeyboardEvents()
	actions = [p.readUserDebugParameter(i) for i in debug_params]
	apply_action(body, actions, motors, power)

p.disconnect()
