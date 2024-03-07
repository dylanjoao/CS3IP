import random

import gymnasium as gym
import pendulum_climb
import torso_climb
from torso_climb.env.torso_climb_env import Reward
import pybullet as p
import time
from stable_baselines3 import PPO, SAC

MOTION = [[6, 5]]
STATEFILE = "./torso_climb/states/state3_55.npz"

env = gym.make('TorsoClimb-v0', render_mode='human', max_ep_steps=600, reward=Reward.NEGATIVE_DIST, motion_path=MOTION, state_file=STATEFILE)
ob, info = env.reset(seed=42)

state = env.reset()
done = False
truncated = False
score = 0
step = 0
pause = False
hold = True

action = [0.0 for i in range(6)]
action += [1.0, 1.0]

# Test each joint
# action = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
# action += [0.0, 0.0]


# ====
# model = SAC.load(path="E:\\Programs\\GymRL\\PyBullet\\CS3IP\\CS3IP\\models\\SAC_375000.zip", device="cuda", env=env)
# vec_env = model.get_env()
# obs = vec_env.reset()
# ====

while True:
    # action = env.action_space.sample()

    if not pause:
        # ====
        # action, _state = model.predict(obs, deterministic=True)
        # obs, reward, done, info = vec_env.step(action)
        # ====
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        step += 1

    # Reset on backspace
    keys = p.getKeyboardEvents()

    if 104 in keys and keys[104] & p.KEY_WAS_TRIGGERED:
        hold = not hold
        action[6] = 1.0 if hold else 0.0
        action[7] = 1.0 if hold else 0.0
        print(f"Hold {hold}")

    if 114 in keys and keys[114] & p.KEY_WAS_TRIGGERED:
        print(f"Score: {score}, Steps {step}")
        done = False
        truncated = False
        pause = False
        score = 0
        step = 0
        env.reset()

    # Pause on space
    if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED:
        pause = not pause
        print("Paused" if pause else "Unpaused")

    if done or truncated:
        pause = True

env.close()
