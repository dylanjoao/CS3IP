import random

import gymnasium as gym
import pendulum_climb
import time
import numpy as np

from pprint import pprint

np.set_printoptions(precision=2, linewidth=100)

env = gym.make('PendulumClimb-v0')
obs, info = env.reset(seed=42)

episode = 10
for episode in range(1, episode + 1):
    state = env.reset()
    done = False
    truncated = False
    score = 0
    step = 0

    while not done and not truncated:
        # action = env.action_space.sample()
        action = 1
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        step += 1
        env.render()
        time.sleep(1 / 240)
        # pprint(obs, compact=True)

    print(f"Episode {episode}, Score: {score}, Steps {step}")

env.close()



