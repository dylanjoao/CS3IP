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
    score = 0

    while not done:
        action = env.action_space.sample()
        ob, reward, done, _, info = env.step(action)
        score += reward
        env.render()
        time.sleep(1 / 240)

    print(f"Episode {episode}, Score: {score}")

env.close()
