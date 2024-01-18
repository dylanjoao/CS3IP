from time import sleep
from typing import Optional

import os
import gymnasium as gym
import numpy as np
import math
import pybullet as p
import pybullet_data

from torso_climb.assets.torso import Torso


class TorsoClimbEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode: Optional[str] = None):
        self.render_mode = render_mode

        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        # action space
        # observation space

        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=np.float32)

        self.np_random, _ = gym.utils.seeding.np_random()

        # configure pybullet GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-90, cameraPitch=0, cameraTargetPosition=[0, 0, 2], physicsClientId=self.client)

    def _get_obs(self):
        return dict()

    def _get_info(self):
        return dict()

    def step(self, action):
        p.stepSimulation(physicsClientId=self.client)

        # Gather information about the env
        ob = self._get_obs()
        info = self._get_info()

        reward = 0

        # Check termination conditions
        terminated = False
        truncated = False

        if self.render_mode == 'human':
            sleep(1 / 240)

        return ob, reward, terminated, truncated, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)

        flags = p.URDF_MAINTAIN_LINK_ORDER + p.URDF_USE_SELF_COLLISION + p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        torso = Torso(client=self.client, pos=[0, 0, 5])

        ob = self._get_obs()
        info = self._get_info()

        return ob, info

    def render(self):
        pass

    def close(self):
        p.disconnect(physicsClientId=self.client)
