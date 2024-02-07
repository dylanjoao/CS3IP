from time import sleep
from typing import Optional

import os
import gymnasium as gym
import numpy as np
import math
import pybullet as p
import pybullet_data

from torso_climb.assets.target import Target
from torso_climb.assets.torso import *
from torso_climb.assets.wall import Wall


class TorsoClimbEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode: Optional[str] = None):
        self.render_mode = render_mode

        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        # action space and observation space
        self.action_space = gym.spaces.Box(-1, 1, (8,), np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(856,), dtype=np.float32)

        self.np_random, _ = gym.utils.seeding.np_random()

        self.torso = None
        self.targets = None
        self.highest_point = float('-inf')

        self.current_stance = []
        self.desired_stance = []

        # configure pybullet GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=-90, cameraPitch=0, cameraTargetPosition=[0, 0, 3],
                                     physicsClientId=self.client)

    # position,
    # orientation,
    # inertial frame pos,
    # linear velocity,
    # angular velocity
    # target position and distance away from each effector
    # effector target hold
    def _get_obs(self):
        obs = []

        states = p.getLinkStates(self.torso.human, linkIndices=self.torso.ordered_joint_indices, computeLinkVelocity=1,
                                 physicsClientId=self.client)
        for state in states:
            worldPos, worldOri, localInertialPos, _, _, _, linearVel, angVel = state
            obs += (worldPos + worldOri + localInertialPos + linearVel + angVel)

        # Find euclid distance between target and each effector
        for target in self.targets:
            target_pos, _ = p.getBasePositionAndOrientation(bodyUniqueId=target.id, physicsClientId=self.client)
            states = p.getLinkStates(self.torso.human, linkIndices=[self.torso.LEFT_HAND, self.torso.RIGHT_HAND],
                                     physicsClientId=self.client)
            effector_distances = []
            for state in states:
                effectorPos, _, _, _, _, _ = state
                dist = np.linalg.norm(np.array(target_pos) - np.array(effectorPos))
                effector_distances.append(dist)
            target_obs = target_pos + tuple(effector_distances)
            obs += target_obs

        obs += (self.torso.lhand_cid, self.torso.rhand_cid)

        # Does it matter what order data is returned?
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        return dict()

    def step(self, action):

        p.stepSimulation(physicsClientId=self.client)

        self.torso.apply_action(action)

        # Gather information about the env
        ob = self._get_obs()
        info = self._get_info()

        reward = self.caclulate_reward()

        # Check termination conditions
        terminated = False
        truncated = False

        if self.render_mode == 'human': sleep(1 / 240)

        return ob, reward, terminated, truncated, info

    # Naderi et al. (2019) Eq 2, A Reinforcement Learning Approach To Synthesizing Climbing Movements
    def calculate_reward_eq1(self):
        pass

    def caclulate_reward(self):
        # Reward if effectors close to hold
        # Reward if moving towards goal
        # Negative rewards?

        left_hand_pos = p.getLinkState(self.torso.human, self.torso.LEFT_HAND, physicsClientId=self.client)[0]
        right_hand_pos = p.getLinkState(self.torso.human, self.torso.RIGHT_HAND, physicsClientId=self.client)[0]

        # Highest point of z-value
        highest_effector = np.max((left_hand_pos[2], right_hand_pos[2]))
        multipler = 0.0
        if highest_effector > self.highest_point:
            multipler = 1.0
            self.highest_point = highest_effector

        return 1.0 * multipler

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 60.,
                                    numSolverIterations=100,
                                    numSubSteps=10,
                                    physicsClientId=self.client)

        flags = p.URDF_MAINTAIN_LINK_ORDER + p.URDF_USE_SELF_COLLISION + p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        wall = Wall(client=self.client, pos=[0.5, 0, 2.5])
        torso = Torso(client=self.client, pos=[-0.25, 0, 0.15])

        self.targets = []
        for i in range(1, 15):  # Vertical
            for j in range(1, 10):  # Horizontal
                self.targets.append(Target(client=self.client, pos=[0.40, (j * 0.25) - 1.25, i * 0.25]))

        self.torso = torso
        ob = self._get_obs()
        info = self._get_info()

        # self.torso.force_attach(self.torso.LEFT_HAND, self.targets[15].id, force=-1)
        # self.torso.force_attach(self.torso.RIGHT_HAND, self.targets[1].id, force=100)

        return np.array(ob, dtype=np.float32), info

    def render(self):
        pass

    def close(self):
        p.disconnect(physicsClientId=self.client)
