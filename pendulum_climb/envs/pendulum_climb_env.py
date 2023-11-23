import os
from time import sleep

import gym
import numpy as np
import math
import pybullet as p
import pybullet_data

from pendulum_climb.assets.pendulum import Pendulum
from pendulum_climb.assets.target import Target


class PendulumClimbEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Add momentum, grasp, release
        self.action_space = gym.spaces.box.Box(
            low=np.array([0, -200], dtype=np.float32),
            high=np.array([2, 200], dtype=np.float32))

        # position[3], orientation[3], velocity, goal[3]
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-float('inf'), -float('inf'), -float('inf'),
                          -float('inf'), -float('inf'), -float('inf'),
                          -float('inf'),
                          -float('inf'), -float('inf'), -float('inf')],
                         dtype=np.float32),
            high=np.array([float('inf'), float('inf'), float('inf'),
                           float('inf'), float('inf'), float('inf'),
                           float('inf'),
                           float('inf'), float('inf'), float('inf')],
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

        dist_to_goal = math.sqrt(((pen_ob[0] - self.goal[0]) ** 2 +
                                  (pen_ob[1] - self.goal[1]) ** 2 +
                                  (pen_ob[2] - self.goal[2]) ** 2))

        reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        self.prev_dist_to_goal = dist_to_goal

        ob = np.array(pen_ob + self.goal, dtype=np.float32)
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
        self.done = False
        self.targets.clear()

        # Targets equally apart
        dist = 2
        for i in range(4):
            target = Target(self.client, [0, 0, i * 2 + dist])
            self.targets.append(target)

        self.targets[0].constraint = p.createConstraint(parentBodyUniqueId=self.pendulum.id,
                                                        parentLinkIndex=0,
                                                        childBodyUniqueId=self.targets[0].id,
                                                        childLinkIndex=-1,
                                                        jointType=p.JOINT_POINT2POINT,
                                                        jointAxis=[0, 0, 0],
                                                        parentFramePosition=[0, 0, 0],
                                                        childFramePosition=[0, 0, 0])
        self.pendulum.top_held = self.targets[0]
        self.pendulum.targets = self.targets

        goal_pos, _ = p.getBasePositionAndOrientation(self.targets[-1].id, self.client)
        self.goal = goal_pos

        # Get observation to return
        ob = self.pendulum.get_observation()

        self.prev_dist_to_goal = math.sqrt(((ob[0] - self.goal[0]) ** 2 +
                                            (ob[1] - self.goal[1]) ** 2 +
                                            (ob[2] - self.goal[2]) ** 2))

        return np.array(ob + self.goal, dtype=np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.client)
