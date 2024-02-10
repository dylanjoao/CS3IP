import math
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

flags = p.URDF_MAINTAIN_LINK_ORDER + p.URDF_USE_SELF_COLLISION
plane = p.loadURDF("plane.urdf")
humanoid = p.loadURDF(current_directory + "/torso_climb/assets/pyb_torso_2.xml", basePosition=[0, 0, 2], baseOrientation=[0.707, 0, 0, 0.707], flags=flags,
                      globalScaling=0.25, useFixedBase=True)

for j in range(p.getNumJoints(humanoid)):
    ji = p.getJointInfo(humanoid, j)
    targetPosition = [0]
    jointType = ji[2]
    if jointType == p.JOINT_SPHERICAL:
        targetPosition = [0, 0, 0, 0]
        p.setJointMotorControlMultiDof(humanoid,
                                       j,
                                       p.POSITION_CONTROL,
                                       targetPosition,
                                       targetVelocity=[0, 0, 0],
                                       positionGain=0,
                                       velocityGain=1,
                                       force=[0, 0, 0])
    if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
        p.setJointMotorControl2(humanoid, j, p.VELOCITY_CONTROL, targetVelocity=0, force=0)


while True:
    p.stepSimulation()
    events = p.getKeyboardEvents()

p.disconnect()
