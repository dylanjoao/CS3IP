import gym
import pendulum_climb
import time


env = gym.make('PendulumClimb-v0')
ob = env.reset()
while True:
    ob, _, done, _ = env.step([0, 20])
    env.render()
    if done:
        ob = env.reset()

    time.sleep(1.0 / 240.0)