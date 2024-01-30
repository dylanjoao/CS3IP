import random

import gymnasium as gym
import pendulum_climb
import torso_climb
import pybullet as p
import time

env = gym.make('TorsoClimb-v0', render_mode='human')
ob, info = env.reset(seed=42)

state = env.reset()
done = False
truncated = False
score = 0
step = 0

action = [0.0 for i in range(6)]

while True:
    # action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    score += reward
    step += 1

    # Reset on backspace
    keys = p.getKeyboardEvents()
    if 65305 in keys and keys[65305]&p.KEY_WAS_TRIGGERED or done or truncated:
        print(f"Score: {score}, Steps {step}")
        done = False
        truncated = False
        score = 0
        step = 0
        env.reset()

env.close()
