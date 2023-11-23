import os
from time import sleep

import gym
import numpy as np
import math
import pybullet as p
import pybullet_data

from pendulum_climb.assets.pendulum import Pendulum


class PendulumClimbEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Add momentum, Grasp, Release
        self.action_space = gym.spaces.box.Box(
            low=np.array([0, -200], dtype=np.float32),
            high=np.array([2, 200], dtype=np.float32))

        # Position, orientation, velocity
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'),
                          -float('inf')], dtype=np.float32),
            high=np.array(
                [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
                dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1 / 30, self.client)

        self.pendulum = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.targets = []

        self.reset()

    def step(self, action):
        # Feed action to the pendulum and get observation of pendulum's state
        self.pendulum.apply_action(action)
        p.stepSimulation()
        pen_ob = self.pendulum.get_observation()
        reward = 10
        ob = np.array(pen_ob, dtype=np.float32)
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)
        # Reload the plane and car
        plane = p.loadURDF("plane.urdf")
        self.pendulum = Pendulum(self.client)

        # Set the goal to a random target
        # x = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
        #      self.np_random.uniform(-9, -5))
        # y = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
        #      self.np_random.uniform(-9, -5))
        # self.goal = (x, y)
        self.done = False
        self.targets.clear()

        # Targets equally apart
        current_directory = os.getcwd()
        for i in range(4):
            target = p.loadURDF(current_directory + "/pendulum_climb/assets/target.urdf",
                                basePosition=[0, 0, i * 2 + 2],
                                useFixedBase=True)
            self.targets.append(target)

        # # Visual element of the goal
        # Goal(self.client, self.goal)

        # Get observation to return
        ob = self.pendulum.get_observation()

        # self.prev_dist_to_goal = math.sqrt(((ob[0] - self.goal[0]) ** 2 +
        #                                     (ob[1] - self.goal[1]) ** 2))
        return np.array(ob, dtype=np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.client)
