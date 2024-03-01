import random

import pybullet as p
import torso_climb
import stable_baselines3 as sb
import gymnasium as gym
import numpy as np
import time

NUM_SAMPLES = 1000


def get_state(bodyIndex, pid):
	# pos[3], ori[4], (jointPos, jointVel), (jointPos, jointVel)...

	pos, ori = p.getBasePositionAndOrientation(bodyIndex, physicsClientId=pid)
	numJoints = p.getNumJoints(bodyIndex, physicsClientId=pid)
	_jointStates = p.getJointStates(bodyIndex, [n for n in range(numJoints)])
	jointStates = [s[:2] for s in _jointStates]

	final_state = []
	final_state += pos
	final_state += ori
	for s in jointStates:
		final_state += s

	return np.array(final_state)


def set_state(bodyIndex, pid, state):
	pos = state[0:3]
	ori = state[3:7]
	numJoints = p.getNumJoints(bodyIndex, physicsClientId=pid)
	joints = [state[(i * 2) + 7:(i * 2) + 9] for i in range(numJoints)]

	p.resetBasePositionAndOrientation(bodyIndex, pos, ori, physicsClientId=pid)
	for joint in range(numJoints):
		p.resetJointState(bodyIndex, joint, joints[joint][0], joints[joint][1], physicsClientId=pid)


env = gym.make("TorsoClimb-v0")
model = sb.PPO.load("../models/stance1_best_model.zip", env=env)
obs = env.reset()[0]

# file = np.load(r"./final_states_1.npz")
# done = False
# info = None
# while True:
# 	action, _state = model.predict(np.array(obs), deterministic=True)
# 	obs, reward, done, truncated, info = env.step(action)
#
# 	keys = p.getKeyboardEvents()
# 	if 114 in keys and keys[114] & p.KEY_WAS_TRIGGERED:
# 		env.reset()
#
# 	if 104 in keys and keys[104] & p.KEY_WAS_TRIGGERED:
# 		rand = random.randint(0, 999)
# 		state = file['arr_0'][rand]
# 		set_state(2, 0, state)

start = time.perf_counter()
saved = 0
states = []
for i in range(NUM_SAMPLES):
	done = False
	truncated = False
	info = None
	while not done and not truncated:
		action, _state = model.predict(np.array(obs), deterministic=True)
		obs, reward, done, truncated, info = env.step(action)

	if info['is_success']:
		states.append(get_state(2, 0))
		saved += 1

	print(f"reset {saved}")
	env.reset()

env.close()
np.savez(r'./out.npz', states)
print(f"Saved {NUM_SAMPLES} samples in {time.perf_counter() - start} seconds")
