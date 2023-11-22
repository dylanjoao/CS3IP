import math
from time import sleep
import pybullet as p
import pybullet_data
import numpy as np
import os

current_directory = os.getcwd()

# Can alternatively pass in p.DIRECT
client = p.connect(p.GUI)
p.setGravity(0, 0, -10, physicsClientId=client)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

plane = p.loadURDF("plane.urdf")
target1 = p.loadURDF(current_directory + "/pendulum_climb/assets/target.urdf", basePosition=[0, 0, 2],
                     useFixedBase=True)
target2 = p.loadURDF(current_directory + "/pendulum_climb/assets/target.urdf", basePosition=[0, 0, 4],
                     useFixedBase=True)
target3 = p.loadURDF(current_directory + "/pendulum_climb/assets/target.urdf", basePosition=[0, 0, 6],
                     useFixedBase=True)
target4 = p.loadURDF(current_directory + "/pendulum_climb/assets/target.urdf", basePosition=[0, 0, 8],
                     useFixedBase=True)
pendulum = p.loadURDF(current_directory + "/pendulum_climb/assets/pendulum.urdf", basePosition=[0, 0, 1])

# Joint links of pendulum
pendulum_indices = [0, 1]

# directionX = p.addUserDebugParameter('DirectionX', -0.5, 0.5, 0)
# directionY = p.addUserDebugParameter('DirectionY', -0.5, 0.5, 0)
momentum = p.addUserDebugParameter('Momentum', -200, 200, 0)

p.setRealTimeSimulation(1)

# Test constraint to hold onto the first target
# JOINT_FIXED is not affected by gravity
constraint_id = p.createConstraint(parentBodyUniqueId=pendulum,
                                   parentLinkIndex=0,
                                   childBodyUniqueId=target1,
                                   childLinkIndex=-1,
                                   jointType=p.JOINT_POINT2POINT,
                                   jointAxis=[0, 0, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
def is_in_range():
    pendulum_pos, _ = p.getBasePositionAndOrientation(pendulum)
    target_pos, _ = p.getBasePositionAndOrientation(target1)
    distance = np.linalg.norm(np.array(pendulum_pos) - np.array(target_pos))
    return distance < 1.0


while True:
    user_momentum = p.readUserDebugParameter(momentum)

    # pos, ori = p.getBasePositionAndOrientation(pendulum)
    # p.applyExternalForce(pendulum, 0, [50, 0, 0], pos, p.WORLD_FRAME)

    p.setJointMotorControl2(bodyIndex=pendulum,
                            jointIndex=0,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=user_momentum)

    p.stepSimulation()
    sleep(1.0 / 240)

    # if is_in_range():
    #     print("true")

p.disconnect()
