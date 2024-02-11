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
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(447,), dtype=np.float32)

        self.np_random, _ = gym.utils.seeding.np_random()

        self.floor = None
        self.wall = None
        self.torso = None
        self.targets = None

        self.effectors = []
        self.current_stance = []
        self.desired_stance = []
        self.motion_path = []
        self.best_dist_to_stance = []

        # configure pybullet GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=-90, cameraPitch=0, cameraTargetPosition=[0, 0, 3], physicsClientId=self.client)

    def step(self, action):

        p.stepSimulation(physicsClientId=self.client)

        self.torso.apply_action(action)
        self.update_stance()

        # Gather information about the env
        ob = self._get_obs()
        info = self._get_info()

        reward = self.calculate_reward_eq1()

        # Check termination conditions
        terminated = False
        truncated = False

        if len(self.motion_path) == 0:
            terminated = True

        if self.render_mode == 'human': sleep(1 / 240)

        return ob, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 60., numSolverIterations=100, numSubSteps=10, physicsClientId=self.client)

        flags = p.URDF_MAINTAIN_LINK_ORDER + p.URDF_USE_SELF_COLLISION + p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        wall = Wall(client=self.client, pos=[0.5, 0, 2.5])
        torso = Torso(client=self.client, pos=[-0.1, 0, 0.2], ori=[0, 0, 0, 1])

        self.targets = []
        for i in range(1, 8):  # Vertical
            for j in range(1, 8):  # Horizontal
                position = [0.40, (j * 0.4) - 1.6, i * 0.4 + 0.0]
                self.targets.append(Target(client=self.client, pos=position))
                position[2] += 0.05
                p.addUserDebugText(text=f"{len(self.targets) - 1}", textPosition=position, textSize=0.7, lifeTime=0.0,
                                   textColorRGB=[0.0, 0.0, 1.0], physicsClientId=self.client)

        self.wall = wall.id
        self.floor = plane
        self.torso = torso
        self.effectors = [self.torso.LEFT_HAND, self.torso.RIGHT_HAND]
        self.current_stance = [-1, -1]
        self.desired_stance = []
        self.motion_path = [[4, 2], [11, 9], [18, 16], [25, 23], [32, 30], [39, 37], [45, 45]]
        self.best_dist_to_stance = [9999, 9999]

        self.desired_stance = self.motion_path.pop(0)

        ob = self._get_obs()
        info = self._get_info()

        for i, v in enumerate(self.desired_stance):
            p.changeVisualShape(objectUniqueId=self.targets[v].id, linkIndex=-1, rgbaColor=[0.0, 0.7, 0.1, 0.75], physicsClientId=self.client)

        # self.torso.force_attach(self.torso.LEFT_HAND, self.targets[11].id, force=100)
        # self.torso.force_attach(self.torso.RIGHT_HAND, self.targets[2].id, force=-1)

        return np.array(ob, dtype=np.float32), info

    # pos, ori, inertial frame pos, linear vel, angular vel, target position and distance away from each effector
    # effector target hold
    # current stance, desired stance, difference in stances, best_dist_to_stance, touching ground and wall
    def _get_obs(self):
        obs = []

        states = p.getLinkStates(self.torso.human, linkIndices=self.torso.ordered_joint_indices, computeLinkVelocity=1, physicsClientId=self.client)
        for state in states:
            worldPos, worldOri, localInertialPos, _, _, _, linearVel, angVel = state
            obs += (worldPos + worldOri + localInertialPos + linearVel + angVel)

        # Find euclid distance between target and each effector
        states = p.getLinkStates(self.torso.human, linkIndices=[self.torso.LEFT_HAND, self.torso.RIGHT_HAND], physicsClientId=self.client)
        # this whole section is bad practice... whatever
        left_hand_state_pos = states[0][0]
        right_hand_state_pos = states[1][0]
        dist_to_desired = [-1, -1]
        for i, target in enumerate(self.targets):
            target_pos, _ = p.getBasePositionAndOrientation(bodyUniqueId=target.id, physicsClientId=self.client)
            effector_distances = []

            dist_lh = np.linalg.norm(np.array(target_pos) - np.array(left_hand_state_pos))
            dist_rh = np.linalg.norm(np.array(target_pos) - np.array(right_hand_state_pos))
            effector_distances.append(dist_lh)
            effector_distances.append(dist_rh)

            target_obs = target_pos + tuple(effector_distances)
            obs += target_obs

        obs += self.current_stance
        obs += self.desired_stance
        obs += [1 if self.current_stance[i] == self.desired_stance[i] else 0 for i in range(len(self.current_stance))]
        obs += self.best_dist_to_stance
        obs += [1 if self.is_touching_body(self.floor) else 0]
        obs += [1 if self.is_touching_body(self.wall) else 0]

        # Does it matter what order data is returned?
        return np.array(obs, dtype=np.float32)

    # Naderi et al. (2019) Eq 2, A Reinforcement Learning Approach To Synthesizing Climbing Movements
    def calculate_reward_eq1(self):
        # Tuning params
        kappa = 0.5
        sigma = 0.5

        states = p.getLinkStates(self.torso.human, linkIndices=[self.torso.LEFT_HAND, self.torso.RIGHT_HAND], physicsClientId=self.client)

        # Summation of distance away from hold
        term_values = [0, 0]
        current_dist_away = [float('inf'), float('inf')]
        for i, effector in enumerate(self.effectors):
            if self.desired_stance[i] == -1:  # what to do here
                continue
            desired_eff_pos = p.getBasePositionAndOrientation(bodyUniqueId=self.targets[self.desired_stance[i]].id, physicsClientId=self.client)[0]
            current_eff_pos = states[i][0]
            distance = np.linalg.norm(np.array(desired_eff_pos) - np.array(current_eff_pos))
            current_dist_away[i] = distance
            reached = 1 if self.current_stance[i] == self.desired_stance[i] else 0

            term_values[i] = kappa * np.exp(-1 * sigma * distance) + reached

        # I(d_t), is the stance closer than ever
        # Note: I use sum instead of comparing individual elements
        is_closer = 1 if np.sum(current_dist_away) < np.sum(self.best_dist_to_stance) else 0

        # on ground might be too unforgiving for this environment
        is_grounded = self.is_touching_body(self.floor)

        if is_closer:
            for i, v in enumerate(self.best_dist_to_stance):
                if current_dist_away[i] < v:
                    self.best_dist_to_stance[i] = current_dist_away[i]

        reward = is_closer * np.sum(term_values) - is_grounded
        return reward

    def update_stance(self):
        self.get_stance_for_effector(0, self.torso.lhand_cid)
        self.get_stance_for_effector(1, self.torso.rhand_cid)

        if self.current_stance == self.desired_stance and len(self.motion_path) != 0:
            new_stance = self.motion_path.pop(0)
            for i, v in enumerate(self.desired_stance):
                p.changeVisualShape(objectUniqueId=self.targets[v].id, linkIndex=-1, rgbaColor=[1.0, 0.0, 0.0, 0.75], physicsClientId=self.client)
            self.desired_stance = new_stance
            for i, v in enumerate(self.desired_stance):
                p.changeVisualShape(objectUniqueId=self.targets[v].id, linkIndex=-1, rgbaColor=[0.0, 0.7, 0.1, 0.75], physicsClientId=self.client)


        torso_pos = np.array(p.getBasePositionAndOrientation(bodyUniqueId=self.torso.human, physicsClientId=self.client)[0])
        torso_pos[1] += 0.15
        torso_pos[2] += 0.20
        p.addUserDebugText(text=f"{self.current_stance}", textPosition=torso_pos, textSize=1, lifeTime=1 / 30,
                           textColorRGB=[1.0, 0.0, 1.0], physicsClientId=self.client)

    def get_stance_for_effector(self, eff_index, eff_cid):
        if eff_cid != -1:
            target_id = p.getConstraintInfo(constraintUniqueId=eff_cid, physicsClientId=self.client)[2]
            for i, target in enumerate(self.targets):
                if target.id == target_id:
                    self.current_stance[eff_index] = i
                    return
        self.current_stance[eff_index] = -1

    def is_touching_body(self, body):
        contact_points = p.getContactPoints(bodyA=self.torso.human, bodyB=body, physicsClientId=self.client)
        return len(contact_points) > 0

    def render(self):
        pass

    def close(self):
        p.disconnect(physicsClientId=self.client)

    def _get_info(self):
        return dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
