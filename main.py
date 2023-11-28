import random

import gymnasium as gym
import pendulum_climb
import time

env = gym.make('PendulumClimb-v0')
ob, info = env.reset(seed=42)

episode = 10
for episode in range(1, episode + 1):
    state = env.reset()
    done = False
    truncated = False
    score = 0
    step = 0

    while not done and not truncated:
        action = env.action_space.sample()
        ob, reward, done, truncated, info = env.step(1)
        score += reward
        step += 1
        env.render()
        time.sleep(1 / 240)

    print(f"Episode {episode}, Score: {score}, Steps {step}")

env.close()
