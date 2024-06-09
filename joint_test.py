import pybullet
import pybullet_data
import pybullet as p
from pybullet_utils import bullet_client
from humanoid_climb.assets.humanoid import Humanoid
from humanoid_climb.assets.robot_util import *

client = bullet_client.BulletClient(connection_mode=p.GUI)
physicsClientId = client._client

p.setGravity(0, 0, -9.8, physicsClientId=physicsClientId)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# SET REAL TIME SIMULATION MESSES UP setJointMotorControl
# p.setRealTimeSimulation(1, physicsClientId)

plane = p.loadURDF("plane.urdf")
torso = p.loadMJCF("./humanoid_climb/assets/humanoid_symmetric.xml")

stuff = addToScene(client, [torso[0]])

debug_params = []

num_bodies = client.getNumBodies()

for j in range(client.getNumJoints(torso[0])):
	jointInfo = client.getJointInfo(torso[0], j)
	joint_name = jointInfo[1]
	part_name = jointInfo[12]

	joint_name = joint_name.decode("utf8")
	part_name = part_name.decode("utf8")
	print(f"{joint_name}, {part_name}")


# for i in range(len(torso.ordered_joints)):
# 	joint = torso.ordered_joints[i]
# 	debug_params.append(p.addUserDebugParameter(torso.motor_names[i], -1, 1, 0.0, physicsClientId))


while True:
	actions = [p.readUserDebugParameter(i, physicsClientId) for i in debug_params]
	actions += (0.0, 0.0, 0.0, 0.0)
	torso.apply_action(actions)
	p.stepSimulation(physicsClientId)