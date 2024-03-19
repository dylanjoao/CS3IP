import time

import gymnasium as gym
import pendulum_climb

env = gym.make('PendulumClimb-v0', render_mode="human")

ob, info = env.reset()
state = env.reset()


while True:
    obs, reward, done, truncated, info = env.step(4)

    if done or truncated:
        env.reset()

    time.sleep(1/240)