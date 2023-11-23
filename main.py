import gym
import pendulum_climb
import time

env = gym.make('PendulumClimb-v0')
ob = env.reset()
while True:
    action = env.action_space.sample()
    ob, reward, done, _ = env.step(action)
    env.render()
    if done:
        ob = env.reset()

    print(action)


    time.sleep(1.0 / 240.0)
