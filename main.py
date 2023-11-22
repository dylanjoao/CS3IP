from time import sleep
import pybullet as p
import pybullet_data
import numpy as np

# Can alternatively pass in p.DIRECT
client = p.connect(p.GUI)
p.setGravity(0, 0, -10, physicsClientId=client)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

plane = p.loadURDF("plane.urdf")
target1 = p.loadURDF("E:/Programs/GymRL/PyBullet/CS3IP/pendulum_climb/assets/target.urdf", basePosition=[0, 0, 2], useFixedBase=True)
target2 = p.loadURDF("E:/Programs/GymRL/PyBullet/CS3IP/pendulum_climb/assets/target.urdf", basePosition=[0, 0, 4], useFixedBase=True)
target3 = p.loadURDF("E:/Programs/GymRL/PyBullet/CS3IP/pendulum_climb/assets/target.urdf", basePosition=[0, 0, 6], useFixedBase=True)
target4 = p.loadURDF("E:/Programs/GymRL/PyBullet/CS3IP/pendulum_climb/assets/target.urdf", basePosition=[0, 0, 8], useFixedBase=True)
pendulum = p.loadURDF("E:/Programs/GymRL/PyBullet/CS3IP/pendulum_climb/assets/pendulum.urdf", basePosition=[0, 0, 1])

pendulum_indices = [0, 1]

direction = p.addUserDebugParameter('DirectionX', -0.5, 0.5, 0)
direction = p.addUserDebugParameter('DirectionY', -0.5, 0.5, 0)
momentum = p.addUserDebugParameter('Momentum', 0, 20, 0)

# Enable real-time simulation
p.setRealTimeSimulation(1)

constraint_id = p.createConstraint(parentBodyUniqueId=pendulum,
                                   parentLinkIndex=-1,
                                   childBodyUniqueId=target1,
                                   childLinkIndex=-1,
                                   jointType=p.JOINT_POINT2POINT,
                                   jointAxis=[0, 0, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])

def is_in_range():
    cylinder_pos, _ = p.getBasePositionAndOrientation(pendulum)
    target_pos, _ = p.getBasePositionAndOrientation(target1)
    distance = np.linalg.norm(np.array(cylinder_pos) - np.array(target_pos))
    return distance < 1.0  # Adjust the range as needed

while True:
    user_throttle = p.readUserDebugParameter(momentum)

    for joint_index in pendulum_indices:
        p.setJointMotorControl2(pendulum, joint_index,
                                p.TORQUE_CONTROL,  # Use torque control
                                force=user_throttle)  # Apply torque based on user input


    p.stepSimulation()
    sleep(1.0 / 240)  # Adjust the sleep duration based on your desired simulation frequency

    if is_in_range():
        print("true")
        


p.disconnect()
