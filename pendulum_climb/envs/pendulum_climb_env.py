import os
from time import sleep
from typing import Optional

import gymnasium as gym
import numpy as np
import math
import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

from pendulum_climb.assets.pendulum import Pendulum
from pendulum_climb.assets.target import Target


class PendulumClimbEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode: Optional[str] = None):

        self.render_mode = render_mode
        if self.render_mode == 'human':
            self._p = BulletClient(p.GUI)
        else:
            self._p = BulletClient(p.DIRECT)

        # +vel, -vel, grasp, release
        self.action_space = gym.spaces.Discrete(8)

        # Holds[2], Position[3], Angle[3], Velocity[3], Distance[1]
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(13,), dtype=np.float32)

        self.np_random, _ = gym.utils.seeding.np_random()

        self.pendulum = None
        self.pendulum_pos = []
        self.goal = None
        self.initial_dist = None
        self.prev_dist = None
        self.current_distance = None
        self.targets = []
        self._max_episode_steps = 1000
        self._elapsed_steps = 0

        # configure pybullet GUI
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self._p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=90, cameraPitch=0,
                                           cameraTargetPosition=[0, 0, 5])
        self._p.setGravity(0, 0, -9.8)

        # Reload the plane and car
        plane = self._p.loadURDF("plane.urdf")
        self.pendulum = Pendulum(self._p, [0, 0, 2.5])

        # Targets equally apart
        dist = 1.0
        for i in range(10):
            target = Target(self._p, [0, 0, i + 1 * 2 + dist])
            self.targets.append(target)

    def _get_obs(self):
        holds, pos, ang, vel, eff1, eff2 = self.pendulum.get_observation()
        self.pendulum_pos = pos
        self.current_distance = np.linalg.norm(np.array(pos) - np.array(self.goal))

        eff1_dist = np.linalg.norm(np.array(eff1) - np.array(self.goal))
        eff2_dist = np.linalg.norm(np.array(eff2) - np.array(self.goal))

        return np.concatenate((holds, pos, ang, vel, (eff1_dist, eff2_dist)), dtype=np.float32)

    def _get_info(self):
        return {"distance:": np.linalg.norm(np.array(self.pendulum_pos) - np.array(self.goal))}

    def step(self, action):
        # Feed action to the pendulum and get observation of pendulum's state
        self.pendulum.apply_action(action)

        self._p.stepSimulation()

        # Gather information about the env
        ob = self._get_obs()
        info = self._get_info()

        # Reward based on distance towards goal
        reward = max(self.prev_dist - self.current_distance, 0)

        # Check termination conditions
        terminated = False
        truncated = False

        if self.prev_dist + 0.05 < self.current_distance:
            terminated = True
        if self.current_distance < 0.05:
            terminated = True
            reward = 100
        elif self.pendulum_pos[2] < 0.8 or self.pendulum_pos[2] > 50:
            terminated = True

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
        self._elapsed_steps += 1

        self.prev_dist = self.current_distance

        return ob, reward, terminated, truncated, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.pendulum.reset_state()
        initial_constraint = self._p.createConstraint(parentBodyUniqueId=self.pendulum.id,
                                                      parentLinkIndex=0,
                                                      childBodyUniqueId=self.targets[0].id,
                                                      childLinkIndex=-1,
                                                      jointType=p.JOINT_POINT2POINT,
                                                      jointAxis=[0, 0, 0],
                                                      parentFramePosition=[0, 0, 0],
                                                      childFramePosition=[0, 0, 0])
        self.pendulum.targets = self.targets
        self.pendulum.top_held = initial_constraint

        self.goal = self.targets[-1].pos

        # Get observation to return
        ob = self._get_obs()

        self.pendulum_pos = self._p.getBasePositionAndOrientation(self.pendulum.id)[0]
        self.initial_dist = np.linalg.norm(np.array(self.pendulum_pos) - np.array(self.goal))
        self.current_distance = self.initial_dist
        self.prev_dist = self.current_distance
        self._elapsed_steps = 0

        info = self._get_info()

        return ob, info

    def render(self, mode='human'):
        pass

    def close(self):
        self._p.disconnect()
