import gym
import pendulum_climb
import time


env = gym.make('PendulumClimb-v0')
ob = env.reset()
while True:
    ob, reward, done, _ = env.step([1, 0])
    env.render()
    if done:
        ob = env.reset()

    print(f"Reward: {reward}")

    time.sleep(1.0 / 240.0)