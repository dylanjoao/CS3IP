import os
import gymnasium as gym
import numpy as np
import math
import pybullet as p
import pybullet_data

from typing import Optional, List
from pybullet_utils.bullet_client import BulletClient
from humanoid_climb.assets.humanoid import Humanoid
from humanoid_climb.assets.target import Target
from humanoid_climb.assets.wall import Wall


class HumanoidClimbEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, motion_path: List[int], render_mode: Optional[str] = None, max_ep_steps: Optional[int] = 602,
                 state_file: Optional[str] = None):
        self.render_mode = render_mode
        self.max_ep_steps = max_ep_steps
        self.motion_path = motion_path
        self.steps = 0

        if self.render_mode == 'human':
            self._p = BulletClient(p.GUI)
        else:
            self._p = BulletClient(p.DIRECT)

        # 17 joint actions + 4 grasp actions
        self.action_space = gym.spaces.Box(-1, 1, (17,), np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.np_random, _ = gym.utils.seeding.np_random()

        self.effectors = []
        self.current_stance = []
        self.desired_stance = []
        self.desired_stance_index = 0
        self.best_dist_to_stance = []

        # configure pybullet GUI
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self._p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=-90, cameraPitch=0, cameraTargetPosition=[0, 0, 3])
        self._p.setGravity(0, 0, -9.8)
        self._p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240., numSolverIterations=100, numSubSteps=10)

        self.floor = self._p.loadURDF("plane.urdf")
        # self.wall = Wall(self._p, pos=[0.48, 0, 2.5]).id
        self.robot = Humanoid(self._p, [0, 0, 1.175], [0, 0, 0, 1], 0.48, None, False)

        self.targets = []
        for i in range(1, 8):  # Vertical
            h_offset = -1.5
            h_spacing = 0.6
            for j in range(1, 5):  # Horizontal
                v_offset = 0.2 * (j & 1) - 0.4
                v_spacing = 0.7
                position = [0.40, (j * h_spacing) + h_offset, i * v_spacing + v_offset]
                self.targets.append(Target(self._p, pos=position))
                position[2] += 0.05
                self._p.addUserDebugText(text=f"{len(self.targets) - 1}", textPosition=position, textSize=0.7, lifeTime=0.0,
                                   textColorRGB=[0.0, 0.0, 1.0])

    def step(self, action):

        self._p.stepSimulation()

        return [0], 0, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot.reset()
        self.steps = 0
        self.desired_stance_index = 0
        self.desired_stance = self.motion_path[self.desired_stance_index]

        ob = self._get_obs()
        info = self._get_info()

        self.robot.force_attach(self.robot.LEFT_FOOT, self.targets[25], force=1000)

        return np.array(ob, dtype=np.float32), info

    def _get_obs(self):
        return [0]

    def _get_info(self):
        return {}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
