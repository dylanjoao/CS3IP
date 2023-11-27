import gymnasium as gym
import pendulum_climb
import time

gymenv = gym.make('PendulumClimb-v0')
env = gym.wrappers.FlattenObservation(gymenv)
ob, info = env.reset(seed=42)
while True:
    action = env.action_space.sample()
    ob, reward, done, _, info = env.step(1)

    env.render()
    if done:
        ob, info = env.reset()

    time.sleep(1.0 / 240.0)
