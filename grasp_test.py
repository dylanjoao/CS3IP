import numpy as np
import pybullet_data
import pybullet as p
from pybullet_utils import bullet_client

from humanoid_climb.assets.humanoid import Humanoid


def draw_x(client, position, size=0.05, color=[1, 0, 0]):
    """
    Draws an X at the given position in the PyBullet simulation.

    Args:
        client (pybullet): PyBullet client instance.
        position (list): List of [x, y, z] coordinates for the center of the X.
        size (float, optional): Size of the X. Defaults to 1.0.
        color (list, optional): RGB color list for the X. Defaults to [1, 0, 0] (red).
    """
    half_size = size / 2.0
    p1 = [position[0] + half_size, position[1] + half_size, position[2]]
    p2 = [position[0] - half_size, position[1] - half_size, position[2]]
    p3 = [position[0] - half_size, position[1] + half_size, position[2]]
    p4 = [position[0] + half_size, position[1] - half_size, position[2]]

    client.addUserDebugLine(p1, p2, color, lifeTime=60)
    client.addUserDebugLine(p3, p4, color, lifeTime=60)

_p = bullet_client.BulletClient(connection_mode=p.GUI)
_p.setGravity(0, 0, -9.8)
_p.setAdditionalSearchPath(pybullet_data.getDataPath())
_p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240., numSubSteps=10)
_p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

plane = _p.loadURDF("plane.urdf")
robot = Humanoid(_p, [0, 0, 2], [0, 0, 0, 1], 0.48, fixedBase=True)
target = Target(_p, [0.5, 0, 1])
params = [_p.addUserDebugParameter(motor.joint_name, -1, +1) for motor in robot.motors]

collisionFilterGroup = 0
collisionFilterMask = 0
_p.setCollisionFilterGroupMask(target.id, -1, 0, 0)
_p.changeVisualShape(target.id, -1, rgbaColor=[1, 0, 0, 0.2])

chest_group = 1
waist_group = 2
left_leg_group = 4
left_arm_group = 8
right_leg_group = 16
right_arm_group = 32

waist_mask = 0x0
left_leg_mask = right_leg_group | left_arm_group | right_arm_group
left_arm_mask = left_leg_group | right_leg_group | right_arm_group | waist_group
right_leg_mask = left_leg_group | left_arm_group | right_arm_group
right_arm_mask = left_leg_group | right_leg_group | left_arm_group | waist_group

col_groups = {"lwaist": [waist_group, waist_mask],
              "pelvis": [waist_group, waist_mask],

              "right_thigh": [right_leg_group, right_leg_mask],
              "right_shin": [right_leg_group, right_leg_mask],
              "right_foot": [right_leg_group, right_leg_mask],

              "left_thigh": [left_leg_group, left_leg_mask],
              "left_shin": [left_leg_group, left_leg_mask],
              "left_foot": [left_leg_group, left_leg_mask],

              "right_upper_arm": [right_arm_group, right_arm_mask],
              "right_lower_arm": [right_arm_group, right_arm_mask],
              "right_hand": [right_arm_group, right_arm_mask],

              "left_upper_arm": [left_arm_group, left_arm_mask],
              "left_lower_arm": [left_arm_group, left_arm_mask],
              "left_hand": [left_arm_group, left_arm_mask]
              }

robot.targets = [target]
for geom in robot.parts:
    _p.changeVisualShape(robot.robot, robot.parts[geom].bodyPartIndex, rgbaColor=[1, 1, 1, 0.3])
    if geom in col_groups:
        _p.setCollisionFilterGroupMask(robot.robot, robot.parts[geom].bodyPartIndex, col_groups[geom][0],
                                       col_groups[geom][1])
        print(f"{geom} set to group {col_groups[geom][0]} & mask {col_groups[geom][1]}")

constraint = -1

eff = "left_foot"

pause = False
while _p.isConnected():

    if not pause:
        actions = [_p.readUserDebugParameter(param) for param in params]
        actions += (1.0, 0.0, 0.0, 0.0)
        robot.apply_action(actions)
        _p.stepSimulation()

    keys = p.getKeyboardEvents()

    # Pause on space
    if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED:
        pause = not pause
        print("Paused" if pause else "Unpaused")

    # R
    if 114 in keys and keys[114] & p.KEY_WAS_TRIGGERED:
        robot.reset()
        _p.removeAllUserDebugItems()
        _p.removeConstraint(userConstraintUniqueId=constraint)
        _p.removeConstraint(userConstraintUniqueId=robot.base_constraint)
        robot.base_constraint = _p.createConstraint(robot.robot, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0, 1], [0, 0, 2])
        constraint = -1
        pause = True

    # H
    if 104 in keys and keys[104] & p.KEY_WAS_TRIGGERED:
        cp = _p.getClosestPoints(target.id, robot.robot, 100.0, -1, robot.parts[eff].bodyPartIndex)[0]
        contactDistance = cp[8]
        positionOnA = cp[5]
        positionOnB = cp[6]
        print(f"{contactDistance}\n{positionOnA}\n{positionOnB}")

        ls = _p.getLinkState(robot.robot, robot.parts[eff].bodyPartIndex)
        print(ls)
        draw_x(_p, ls[0])

    # J
    if 106 in keys and keys[106] & p.KEY_WAS_TRIGGERED:
        _p.removeConstraint(userConstraintUniqueId=robot.base_constraint)

    cp = _p.getClosestPoints(target.id, robot.robot, 100.0, -1, robot.parts[eff].bodyPartIndex)[0]
    contactDistance = cp[8]

    if contactDistance < 0.0:

        ls = _p.getLinkState(robot.robot, robot.parts[eff].bodyPartIndex)

        positionOnA = cp[5]
        positionOnB = cp[6]
        contactNormalOnB = cp[7]
        # _p.addUserDebugLine(target.pos, positionOnB, [0., 1., 0.], lifeTime=15.0)

        if constraint != -1:
            continue

        # _p.addUserDebugPoints((positionOnA, positionOnB), ([1, 0, 0], [0, 1, 0]), lifeTime=15.0, pointSize=1.0)
        # draw_x(_p, positionOnA, size=0.01, color=[1, 0, 0])
        # draw_x(_p, positionOnB, size=0.01, color=[0, 1, 0])
        current_pos = robot.parts[eff].current_position()
        parentPos = [0, 0, 0]
        childPos = np.subtract(current_pos, target.pos)
        parentOri = [0, 0, 0, 1]
        childOri = [0, 0, 0, 1]
        jointAxis = [0, 0, 0]
        constraint = _p.createConstraint(parentBodyUniqueId=robot.robot,
                                         parentLinkIndex=robot.parts[eff].bodyPartIndex,
                                         childBodyUniqueId=target.id, childLinkIndex=-1,
                                         jointType=p.JOINT_POINT2POINT, jointAxis=jointAxis,
                                         parentFramePosition=parentPos, childFramePosition=childPos,
                                         parentFrameOrientation=parentOri, childFrameOrientation=childOri)
        _p.changeConstraint(userConstraintUniqueId=constraint, maxForce=2500)

        ci = _p.getConstraintInfo(constraint)

        draw_x(_p, childPos)

        print(ci[6])
        print(ci[7])

        # draw_x(_p, parentPos, size=0.05, color=[1, 0, 0])
        # draw_x(_p, childPos, size=0.05, color=[0, 1, 0])
        print(f"Created constraint {constraint}")

_p.disconnect()
