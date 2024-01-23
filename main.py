import random

import gymnasium as gym
import pendulum_climb
import torso_climb
import time

env = gym.make('TorsoClimb-v0', render_mode='human')
ob, info = env.reset(seed=42)

episode = 10
for episode in range(1, episode + 1):
    state = env.reset()
    done = False
    truncated = False
    score = 0
    step = 0

    action = [0.0 for i in range(6)]
    while not done and not truncated:
        # action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        step += 1
        

    print(f"Episode {episode}, Score: {score}, Steps {step}")

env.close()
