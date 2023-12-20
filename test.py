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
p.setGravity(0, 0, -10, physicsClientId=client)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

plane = p.loadURDF("plane.urdf")
target1 = p.loadURDF(current_directory + "/pendulum_climb/assets/target.urdf", basePosition=[0, 0, 3],
                     useFixedBase=True)
target3 = p.loadURDF(current_directory + "/pendulum_climb/assets/target.urdf", basePosition=[0, 0, 5],
                     useFixedBase=True)
target4 = p.loadURDF(current_directory + "/pendulum_climb/assets/target.urdf", basePosition=[0, 0, 7],
                     useFixedBase=True)
# pendulum = p.loadURDF(current_directory + "/pendulum_climb/assets/pendulum.urdf", basePosition=[0, 0, 1])
pendulum = Pendulum(client, [0, 0, 0.5])

# Joint links of pendulum
pendulum_indices = [0]

# directionX = p.addUserDebugParameter('DirectionX', -0.5, 0.5, 0)
# directionY = p.addUserDebugParameter('DirectionY', -0.5, 0.5, 0)
momentumX = p.addUserDebugParameter('MomentumX', -200, 200, 0)
momentumY = p.addUserDebugParameter('MomentumY', -200, 200, 0)

# p.setRealTimeSimulation(1)

# Test constraint to hold onto the first target
# JOINT_FIXED is not affected by gravity
constraint_id = p.createConstraint(parentBodyUniqueId=pendulum.id,
                                   parentLinkIndex=0,
                                   childBodyUniqueId=target1,
                                   childLinkIndex=-1,
                                   jointType=p.JOINT_POINT2POINT,
                                   jointAxis=[0, 0, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])

while True:
    user_momentumX = p.readUserDebugParameter(momentumX)
    user_momentumY = p.readUserDebugParameter(momentumY)

    p.applyExternalTorque(objectUniqueId=pendulum.id,
                          linkIndex=0,
                          torqueObj=[user_momentumX, user_momentumY, 0.0],
                          flags=p.LINK_FRAME)

    # p.applyExternalForce(objectUniqueId=pendulum.id,
    #                      linkIndex=0,
    #                      forceObj=[user_momentumX, user_momentumY, 0.0],
    #                      posObj=[0.0, 0.0, 0.0],
    #                      flags=p.LINK_FRAME)

    # p.setJointMotorControl2(bodyIndex=pendulum.id,
    #                         jointIndex=1,
    #                         controlMode=p.VELOCITY_CONTROL,
    #                         targetVelocity=user_momentum1)
    #
    # p.setJointMotorControl2(bodyIndex=pendulum.id,
    #                         jointIndex=0,
    #                         controlMode=p.VELOCITY_CONTROL,
    #                         targetVelocity=user_momentum2)

    p.stepSimulation()
    # sleep(1.0 / 1024)

p.disconnect()
