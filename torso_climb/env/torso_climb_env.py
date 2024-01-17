from time import sleep
from typing import Optional

import os
import gymnasium as gym
import numpy as np
import math
import pybullet as p
import pybullet_data


class TorsoClimbEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode: Optional[str] = None):

        self.render_mode = render_mode

        self.client = p.connect(p.GUI if self.render_mode == 'human' else p.DIRECT)
        print('human' if self.render_mode == 'human' else 'non-human')

        # action space
        # observation space


        self.np_random, _ = gym.utils.seeding.np_random()

        # configure pybullet GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[0, 0, 5])

    def _get_obs(self):
        return dict()

    def _get_info(self):
        return dict()

    def step(self, action):

        p.stepSimulation()

        # Gather information about the env
        ob = self._get_obs()
        info = self._get_info()

        reward = 0

        # Check termination conditions
        terminated = False
        truncated = False

        if self.render_mode == 'human':
            sleep(1/240)

        return ob, reward, terminated, truncated, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        plane = p.loadURDF("plane.urdf")

        ob = self._get_obs()
        info = self._get_info()

        return ob, info

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)
