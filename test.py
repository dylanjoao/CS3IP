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

# flags = p.URDF_MAINTAIN_LINK_ORDER + p.URDF_USE_SELF_COLLISION
flags = p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
plane = p.loadURDF("plane.urdf")
# humanoid = p.loadURDF(current_directory + "/torso_climb/assets/pyb_torso_2.xml", basePosition=[0, 0, 2], baseOrientation=[0.707, 0, 0, 0.707], flags=flags, globalScaling=0.25, useFixedBase=True)
humanoid = p.loadURDF(current_directory + "/torso_climb/assets/pyb_torso.xml", basePosition=[0, 0, 2], flags=flags, useFixedBase=True)
# humanoid = p.loadMJCF(current_directory + "/torso_climb/assets/mjcf_torso_2.xml")

# for j in range(p.getNumJoints(humanoid)):
#     ji = p.getJointInfo(humanoid, j)
#     targetPosition = [0]
#     jointType = ji[2]
#     if jointType == p.JOINT_SPHERICAL:
#         targetPosition = [0, 0, 0, 0]
#         p.setJointMotorControlMultiDof(humanoid,
#                                        j,
#                                        p.POSITION_CONTROL,
#                                        targetPosition,
#                                        targetVelocity=[0, 0, 0],
#                                        positionGain=0,
#                                        velocityGain=1,
#                                        force=[0, 0, 0])
#     if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
#         p.setJointMotorControl2(humanoid, j, p.VELOCITY_CONTROL, targetVelocity=0, force=0)


while True:
    p.stepSimulation()
    events = p.getKeyboardEvents()

    width = 512
    height = 512
    fov = 60
    aspect = width / height
    near = 0.02
    far = 100
    cameraDistance=4
    cameraYaw=-90
    cameraPitch=0
    cameraTargetPosition=[0, 0, 3]
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
			cameraTargetPosition=cameraTargetPosition,
			distance=cameraDistance,
			yaw=cameraYaw,
			pitch=cameraPitch,
			roll=0,
			upAxisIndex=2)
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Get depth values using Tiny renderer
    images = p.getCameraImage(width,
                            height,
                            view_matrix,
                            projection_matrix,
                            shadow=True,
                            renderer=p.ER_TINY_RENDERER)
    rgb_tiny = np.reshape(images[2], (height, width, 4)) * 1. / 255.

    # time.sleep(1/144)

p.disconnect()
