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

    def __init__(self, render_mode: Optional[str] = None, max_ep_steps: Optional[int] = 1000):
        self.render_mode = render_mode
        self.max_ep_steps = max_ep_steps
        self.steps = 0

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

        # INFO DATA
        self.steps_till_first_grasp_left_hand = -1
        self.steps_till_first_grasp_right_hand = -1
        #

        # configure pybullet GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=-90, cameraPitch=0, cameraTargetPosition=[0, 0, 3], physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 60., numSolverIterations=100, numSubSteps=10, physicsClientId=self.client)

        plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        wall = Wall(client=self.client, pos=[0.5, 0, 2.5])
        torso = Torso(client=self.client, pos=[-0.1, 0, 0.20], ori=[0, 0, 0, 1])

        self.wall = wall.id
        self.floor = plane
        self.torso = torso
        self.effectors = [self.torso.LEFT_HAND, self.torso.RIGHT_HAND]

        self.targets = []
        for i in range(1, 8):  # Vertical
            for j in range(1, 8):  # Horizontal
                position = [0.40, (j * 0.4) - 1.6, i * 0.4 + 0.0]
                self.targets.append(Target(client=self.client, pos=position))
                position[2] += 0.05
                p.addUserDebugText(text=f"{len(self.targets) - 1}", textPosition=position, textSize=0.7, lifeTime=0.0,
                                   textColorRGB=[0.0, 0.0, 1.0], physicsClientId=self.client)

        # INFO DATA
        self.steps_till_first_grasp_left_hand = -1
        self.steps_till_first_grasp_right_hand = -1
        #

    def step(self, action):

        p.stepSimulation(physicsClientId=self.client)

        self.torso.apply_action(action)
        self.update_stance()

        # Gather information about the env
        ob = self._get_obs()
        info = self._get_info()

        reward = self.calculate_reward_eq1()

        # Check termination conditions
        terminated = self.terminate_check()
        truncated = self.truncate_check()

        if self.render_mode == 'human': sleep(1 / 60)

        return ob, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.torso.reset_state()
        self.steps = 0
        self.current_stance = [-1, -1]
        self.motion_path = [[3, 3], [10, 10]]
        self.desired_stance = self.motion_path.pop(0)
        self.best_dist_to_stance = self.get_distance_from_desired_stance()

        ob = self._get_obs()
        info = self._get_info()

        for i, target in enumerate(self.targets):
            colour = [0.0, 0.7, 0.1, 0.75] if i in self.desired_stance else [1.0, 0, 0, 0.75]
            p.changeVisualShape(objectUniqueId=target.id, linkIndex=-1, rgbaColor=colour, physicsClientId=self.client)

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

    def terminate_check(self):
        terminated = False

        # Check if completed path
        if len(self.motion_path) == 0:
            terminated = True

        # Check if anything but torso is touching ground
        for i, v in enumerate(self.torso.ordered_joint_indices):
            if self.is_touching_body(self.floor, v):
                terminated = True
                break

        return terminated

    def truncate_check(self):
        self.steps += 1
        truncated = True if self.steps >= self.max_ep_steps else False
        return truncated

    # Naderi et al. (2019) Eq 2, A Reinforcement Learning Approach To Synthesizing Climbing Movements
    def calculate_reward_eq1(self):
        # Tuning params
        kappa = 0.6
        sigma = 0.5

        # Summation of distance away from hold
        sum_values = [0, 0]
        current_dist_away = self.get_distance_from_desired_stance()
        for i, effector in enumerate(self.effectors):
            distance = current_dist_away[i]
            reached = 1 if self.current_stance[i] == self.desired_stance[i] else 0
            sum_values[i] = kappa * np.exp(-1 * sigma * distance) + reached

        # I(d_t), is the stance closer than ever
        # individually check if the distance on both hand is closer than before
        is_closer = True
        difference_closer = 0
        for i, best_dist_away in enumerate(self.best_dist_to_stance):
            difference = best_dist_away - current_dist_away[i]
            if difference <= 0:  # not closer
                is_closer = False
                difference_closer -= difference

        if is_closer:
            for i, best_dist_away in enumerate(self.best_dist_to_stance):
                if current_dist_away[i] < best_dist_away:
                    self.best_dist_to_stance[i] = current_dist_away[i]

        # positive reward if closer, otherwise small penalty based on difference away
        reward = is_closer * np.sum(sum_values) - 0.1 * difference_closer
        self.visualise_reward(reward, -2, 2)

        return reward

    def update_stance(self):
        self.get_stance_for_effector(0, self.torso.lhand_cid)
        self.get_stance_for_effector(1, self.torso.rhand_cid)

        # Check if stance complete
        if self.current_stance == self.desired_stance and len(self.motion_path) != 0:
            new_stance = self.motion_path.pop(0)

            for i, v in enumerate(self.desired_stance):
                p.changeVisualShape(objectUniqueId=self.targets[v].id, linkIndex=-1, rgbaColor=[1.0, 0.0, 0.0, 0.75], physicsClientId=self.client)
            self.desired_stance = new_stance
            for i, v in enumerate(self.desired_stance):
                p.changeVisualShape(objectUniqueId=self.targets[v].id, linkIndex=-1, rgbaColor=[0.0, 0.7, 0.1, 0.75], physicsClientId=self.client)

            # Reset best_dist
            self.best_dist_to_stance = self.get_distance_from_desired_stance()

        if self.render_mode == 'human':
            torso_pos = np.array(p.getBasePositionAndOrientation(bodyUniqueId=self.torso.human, physicsClientId=self.client)[0])
            torso_pos[1] += 0.15
            torso_pos[2] += 0.35
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

    def get_distance_from_desired_stance(self):
        dist_away = [float('inf'), float('inf')]
        states = p.getLinkStates(self.torso.human, linkIndices=[self.torso.LEFT_HAND, self.torso.RIGHT_HAND], physicsClientId=self.client)
        for i, effector in enumerate(self.effectors):
            if self.desired_stance[i] == -1:  # what to do here
                continue

            desired_eff_pos = p.getBasePositionAndOrientation(bodyUniqueId=self.targets[self.desired_stance[i]].id, physicsClientId=self.client)[0]
            current_eff_pos = states[i][0]
            distance = np.abs(np.linalg.norm(np.array(desired_eff_pos) - np.array(current_eff_pos)))
            dist_away[i] = distance
        return dist_away

    def is_touching_body(self, body, link_indexA=-1):
        contact_points = p.getContactPoints(bodyA=self.torso.human, linkIndexA=link_indexA, bodyB=body, physicsClientId=self.client)
        return len(contact_points) > 0

    def render(self):
        pass

    def close(self):
        p.disconnect(physicsClientId=self.client)

    #
    def _get_info(self):
        info = dict()

        # Check if it reached
        if self.steps_till_first_grasp_left_hand == -1:
            if self.current_stance[0] == self.desired_stance[0]:
                self.steps_till_first_grasp_left_hand = self.steps
        if self.steps_till_first_grasp_right_hand == -1:
            if self.current_stance[1] == self.desired_stance[1]:
                self.steps_till_first_grasp_right_hand = self.steps

        info['steps_till_first_hold_reached_lh'] = self.steps_till_first_grasp_left_hand
        info['steps_till_first_hold_reached_rh'] = self.steps_till_first_grasp_right_hand
        return info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def visualise_reward(self, reward, min, max):
        if self.render_mode != 'human': return
        value = np.clip(reward, min, max)
        normalized_value = (value - min) / (max - min) * (1 - 0) + 0
        colour = [0.0, normalized_value / 1.0, 0.0, 1.0] if reward > 0.0 else [normalized_value / 1.0, 0.0, 0.0, 1.0]
        p.changeVisualShape(objectUniqueId=self.torso.human, linkIndex=-1, rgbaColor=colour, physicsClientId=self.client)
