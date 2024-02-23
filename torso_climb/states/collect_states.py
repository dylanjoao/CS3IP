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
	numJoints = p.getNumJoints(bodyIndex, physicsClientId=pid)
	p.resetBasePositionAndOrientation(bodyIndex, state[0], state[1], physicsClientId=pid)
	for joint in range(numJoints):
		p.resetJointState(bodyIndex, joint, state[2][joint][0], state[2][joint][1], physicsClientId=pid)


env = gym.make("TorsoClimb-v0", render_mode=None)
model = sb.PPO.load("../models/stance1_best_model.zip", env=env)
obs = env.reset()[0]

start = time.perf_counter()
saved = 0
states = []
for i in range(NUM_SAMPLES):
	done = False
	info = None
	while not done:
		action, _state = model.predict(np.array(obs), deterministic=True)
		obs, reward, done, truncated, info = env.step(action)

	if info['is_success']:
		states.append(get_state(2, 0))
		saved += 1

	env.reset()

env.close()
np.savez(r'./out.npz', states)
print(f"Saved {NUM_SAMPLES} samples in {time.perf_counter() - start} seconds")
