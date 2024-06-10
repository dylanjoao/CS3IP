import os
from pprint import pprint

import gymnasium as gym
import numpy as np
import math
import pybullet as p
import pybullet_data
import json

from typing import Optional, List
from pybullet_utils.bullet_client import BulletClient
from humanoid_climb.assets.humanoid import Humanoid
from humanoid_climb.env.climbing_config import ClimbingConfig
from humanoid_climb.assets.asset import Asset


class HumanoidClimbEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, config, render_mode: Optional[str] = None, max_ep_steps: Optional[int] = 602, state_file: Optional[str] = None):

        self.config = config

        self.render_mode = render_mode
        self.max_ep_steps = max_ep_steps
        self.steps = 0

        self.motion_path = [self.config.stance_path[stance]['desired_holds'] for stance in self.config.stance_path]
        self.motion_exclude_targets = [self.config.stance_path[stance]['ignore_holds'] for stance in self.config.stance_path]
        self.action_override = [self.config.stance_path[stance]['force_attach'] for stance in self.config.stance_path]

        self.init_from_state = False if state_file is None else True
        self.state_file = state_file

        if self.render_mode == 'human':
            self._p = BulletClient(p.GUI)
        else:
            self._p = BulletClient(p.DIRECT)

        # 17 joint actions + 4 grasp actions
        self.action_space = gym.spaces.Box(-1, 1, (21,), np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(306,), dtype=np.float32)

        self.np_random, _ = gym.utils.seeding.np_random()

        self.current_stance = []
        self.desired_stance = []
        self.desired_stance_index = 0
        self.best_dist_to_stance = []

        # configure pybullet GUI
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self._p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=-90, cameraPitch=0, cameraTargetPosition=[0, 0, 3])
        self._p.setGravity(0, 0, -9.8)
        self._p.setPhysicsEngineParameter(fixedTimeStep= self.config.timestep_interval, numSubSteps=self.config.timestep_per_action)

        self.floor = self._p.loadURDF("plane.urdf")
        self.floor = Asset(self._p, self.config.plane)
        self.wall = Asset(self._p, self.config.surface)
        self.climber = Humanoid(self._p, self.config.climber)

        self.debug_stance_text = self._p.addUserDebugText(text=f"", textPosition=[0, 0, 0], textSize=1, lifeTime=0.1, textColorRGB=[1.0, 0.0, 1.0])

        self.targets = dict()
        for key in self.config.holds:
            self.targets[key] = Asset(self._p, self.config.holds[key])
            self._p.addUserDebugText(text=key, textPosition=self.targets[key].body.initialPosition, textSize=0.7, lifeTime=0.0, textColorRGB=[0.0, 0.0, 1.0])

        self.climber.targets = self.targets

    def step(self, action):

        self._p.stepSimulation()
        self.steps += 1

        self.climber.apply_action(action, self.action_override[self.desired_stance_index])
        self.update_stance()

        ob = self._get_obs()
        info = self._get_info()

        # reward = self.calculate_reward_eq1()
        reward = self.calculate_reward_negative_distance()
        reached = self.check_reached_stance()

        terminated = self.terminate_check()
        truncated = self.truncate_check()

        return ob, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.climber.reset()
        self.climber.exclude_targets = self.motion_exclude_targets[0]
        # if self.init_from_state: self.robot.initialise_from_state()
        self.steps = 0
        self.current_stance = [-1, -1, -1, -1] # TODO
        self.desired_stance_index = 0
        self.desired_stance = self.motion_path[0]
        self.best_dist_to_stance = self.get_distance_from_desired_stance()

        ob = self._get_obs()
        info = self._get_info()

        for key in self.targets:
            colour = [0.0, 0.7, 0.1, 0.75] if key in self.desired_stance else [1.0, 0, 0, 0.75]
            self._p.changeVisualShape(objectUniqueId=self.targets[key].id, linkIndex=-1, rgbaColor=colour)

        return np.array(ob, dtype=np.float32), info

    def calculate_reward_negative_distance(self):
        current_dist_away = self.get_distance_from_desired_stance()

        is_closer = 1 if np.sum(current_dist_away) < np.sum(self.best_dist_to_stance) else 0
        if is_closer: self.best_dist_to_stance = current_dist_away.copy()

        reward = np.clip(-1 * np.sum(current_dist_away), -2, float('inf'))
        # reward += 1000 if self.current_stance == self.desired_stance else 0
        if self.is_on_floor():
            reward += (self.max_ep_steps - self.steps) * -2

        # self.visualise_reward(reward, -6, 0)

        return reward

    def calculate_reward_eq1(self):
        # Tuning params
        kappa = 0.6
        sigma = 0.5

        # Summation of distance away from hold
        sum_values = [0, 0, 0, 0]
        current_dist_away = self.get_distance_from_desired_stance()
        for i, effector in enumerate(self.climber.effectors):
            distance = current_dist_away[i]
            reached = 1 if self.current_stance[i] == self.desired_stance[i] else 0
            sum_values[i] = kappa * np.exp(-1 * sigma * distance) + reached

        # I(d_t), is the stance closer than ever
        is_closer = True
        difference_closer = 0

        # compare sum of values instead of individual values
        if np.sum(current_dist_away) > np.sum(self.best_dist_to_stance):
            is_closer = False
            difference_closer = np.sum(self.best_dist_to_stance) - np.sum(current_dist_away)

        if is_closer:
            # self.best_dist_to_stance = current_dist_away.copy()
            for i, best_dist_away in enumerate(self.best_dist_to_stance):
                if current_dist_away[i] < best_dist_away:
                    self.best_dist_to_stance[i] = current_dist_away[i]

        # positive reward if closer, otherwise small penalty based on difference away
        reward = is_closer * np.sum(sum_values) + 0.8 * difference_closer
        reward += 3000 if self.current_stance == self.desired_stance else 0
        if self.is_on_floor():
            reward = -3000

        self.visualise_reward(reward, -2, 2)

        return reward

    def check_reached_stance(self):
        reached = False

        # Check if stance complete
        if self.current_stance == self.desired_stance:
            reached = True

            self.desired_stance_index += 1
            if self.desired_stance_index > len(self.motion_path) - 1: return

            new_stance = self.motion_path[self.desired_stance_index]
            self.climber.exclude_targets = self.motion_exclude_targets[self.desired_stance_index]

            # Reset current desired target colours to red
            for key in self.desired_stance:
                if key == -1: continue
                self._p.changeVisualShape(objectUniqueId=self.targets[key].id, linkIndex=-1,
                                          rgbaColor=[1.0, 0.0, 0.0, 0.75])
            self.desired_stance = new_stance

            # Set new desired targets to green
            for key in self.desired_stance:
                if key == -1: continue
                self._p.changeVisualShape(objectUniqueId=self.targets[key].id, linkIndex=-1,
                                          rgbaColor=[0.0, 0.7, 0.1, 0.75])

            # Reset best_dist
            self.best_dist_to_stance = self.get_distance_from_desired_stance()

        return reached

    def update_stance(self):
        self.current_stance = self.climber.effector_attached_to

        if self.render_mode == 'human':
            torso_pos = self.climber.robot_body.current_position()
            torso_pos[1] += 0.15
            torso_pos[2] += 0.35
            self.debug_stance_text = self._p.addUserDebugText(text=f"{self.current_stance}", textPosition=torso_pos,
                                                              textSize=1, lifeTime=0.1, textColorRGB=[1.0, 0.0, 1.0],
                                                              replaceItemUniqueId=self.debug_stance_text)

    def get_distance_from_desired_stance(self):
        effector_count = len(self.climber.effectors)
        dist_away = [float('inf') for _ in range(effector_count)]
        effector_positions = [effector.get_position() for effector in self.climber.effectors]

        for eff_index in range(effector_count):
            if self.desired_stance[eff_index] == -1:
                dist_away[eff_index] = 0
                continue

            desired_hold_pos = self.targets[self.desired_stance[eff_index]].body.get_position()
            current_eff_pos = effector_positions[eff_index]
            distance = np.abs(np.linalg.norm(np.array(desired_hold_pos) - np.array(current_eff_pos)))
            dist_away[eff_index] = distance
        return dist_away

    def terminate_check(self):
        terminated = False

        if self.desired_stance_index > len(self.motion_path)-1:
            terminated = True

        if self.is_on_floor():
            terminated = True

        return terminated

    def truncate_check(self):
        truncated = True if self.steps >= self.max_ep_steps else False
        return truncated

    def _get_obs(self):
        obs = []

        states = self._p.getLinkStates(self.climber.robot,
                                       linkIndices=[joint.jointIndex for joint in self.climber.ordered_joints],
                                       computeLinkVelocity=1)

        for state in states:
            worldPos, worldOri, localInertialPos, _, _, _, linearVel, angVel = state
            obs.extend(worldPos + worldOri + localInertialPos + linearVel + angVel)

        eff_positions = [eff.current_position() for eff in self.climber.effectors]
        for i, c_stance in enumerate(self.desired_stance):
            if c_stance == -1:
                obs.extend([-1, -1, -1, 0]) # Dummy values for free limb
                continue

            eff_target = self.targets[c_stance]
            dist = np.linalg.norm(np.array(eff_target.body.initialPosition) - np.array(eff_positions[i]))

            target_obs = eff_target.body.initialPosition
            obs.extend(eff_target.body.initialPosition)
            obs.append(dist)

        obs.extend(-1 if k == -1 else self.targets[k].id for k in self.current_stance)
        obs.extend(-1 if k == -1 else self.targets[k].id for k in self.desired_stance)
        obs.extend([1 if self.current_stance[i] == self.desired_stance[i] else 0 for i in range(len(self.current_stance))])
        obs.extend(self.best_dist_to_stance)
        obs.append(1 if self.is_touching_body(self.floor.id) else 0)
        obs.append(1 if self.is_touching_body(self.wall.id) else 0)

        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        info = dict()

        success = True if self.current_stance == self.desired_stance else False
        info['is_success'] = success

        return info

    def is_on_floor(self):
        touching_floor = False
        floor_contact = self._p.getContactPoints(bodyA=self.climber.robot, bodyB=self.floor.id)
        for i in range(len(floor_contact)):
            contact_body = floor_contact[i][3]
            # TODO
            exclude_list = [self.climber.parts["left_foot"].bodyPartIndex, self.climber.parts["right_foot"].bodyPartIndex]
            if contact_body not in exclude_list:
                touching_floor = True
                break

        return touching_floor

    def is_touching_body(self, bodyB):
        contact_points = self._p.getContactPoints(bodyA=self.climber.robot, bodyB=bodyB)
        return len(contact_points) > 0

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def visualise_reward(self, reward, min, max):
        if self.render_mode != 'human': return
        value = np.clip(reward, min, max)
        normalized_value = (value - min) / (max - min) * (1 - 0) + 0
        colour = [0.0, normalized_value / 1.0, 0.0, 1.0] if reward > 0.0 else [normalized_value / 1.0, 0.0, 0.0, 1.0]
        self._p.changeVisualShape(objectUniqueId=self.climber.robot, linkIndex=-1, rgbaColor=colour)
