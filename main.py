import random

import gymnasium as gym
import pendulum_climb
import torso_climb
import pybullet as p
import time
from stable_baselines3 import PPO, SAC

env = gym.make('TorsoClimb-v0', render_mode='human')
ob, info = env.reset(seed=42)

state = env.reset()
done = False
truncated = False
score = 0
step = 0
pause = False

action = [0.0 for i in range(4)]
action += [1.0, 1.0]

# Test each joint
# action = [0.0, 0.0, 0.0, 0.0]
# action += [0.0, 0.0]


# ====
# model = SAC.load(path="E:\\Programs\\GymRL\\PyBullet\\CS3IP\\CS3IP\\models\\SAC_375000.zip", device="cuda", env=env)
# vec_env = model.get_env()
# obs = vec_env.reset()
# ====

# Pause on end
# Require backspace to restart
# Print information on end or restart


while True:
    # action = env.action_space.sample()

    if not pause:
        # ====
        # action, _state = model.predict(obs, deterministic=True)
        # obs, reward, done, info = vec_env.step(action)
        # ====
        obs, reward, done, truncated, info = env.step(action)
        end_obs = obs[-8:]
        # print(f"{end_obs[0:2]} {end_obs[2:4]} {end_obs[4:6]} {end_obs[6:8]}")

        score += reward
        step += 1

    # Reset on backspace
    keys = p.getKeyboardEvents()
    if 65305 in keys and keys[65305] & p.KEY_WAS_TRIGGERED:
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
