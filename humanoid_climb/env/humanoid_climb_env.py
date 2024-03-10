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
        # self._p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240., numSolverIterations=100, numSubSteps=10)

        self.floor = self._p.loadURDF("plane.urdf")
        self.wall = Wall(self._p, pos=[0.48, 0, 2.5]).id
        self.robot = Humanoid(self._p, [0, 0, 1.175], [0, 0, 0, 1], 0.48, None, False)

        self.targets = []
        for i in range(1, 8):  # Vertical
            h_offset = -1.5
            h_spacing = 0.6
            for j in range(1, 5):  # Horizontal
                v_offset = 0.2 * (j & 1) - 0.4
                v_spacing = 0.65
                position = [0.40, (j * h_spacing) + h_offset, i * v_spacing + v_offset]
                self.targets.append(Target(self._p, pos=position))
                position[2] += 0.05
                self._p.addUserDebugText(text=f"{len(self.targets) - 1}", textPosition=position, textSize=0.7, lifeTime=0.0,
                                   textColorRGB=[0.0, 0.0, 1.0])

        self.robot.set_targets(self.targets)

    def step(self, action):

        self._p.stepSimulation()
        self.steps += 1

        self.robot.apply_action(action)
        self.update_stance()

        ob = self._get_obs()
        info = self._get_info()

        reward = self.calculate_reward_negative_distance()
        reached = self.check_reached_stance()


        terminated = self.terminate_check()
        truncated = self.truncate_check()

        return ob, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot.reset()
        self.steps = 0
        self.current_stance = [-1, -1, -1, -1]
        self.desired_stance_index = 0
        self.desired_stance = self.motion_path[self.desired_stance_index]
        self.best_dist_to_stance = self.get_distance_from_desired_stance()

        ob = self._get_obs()
        info = self._get_info()

        # self.robot.force_attach(self.robot.LEFT_FOOT, self.targets[25], force=1000)

        for i, target in enumerate(self.targets):
            colour = [0.0, 0.7, 0.1, 0.75] if i in self.desired_stance else [1.0, 0, 0, 0.75]
            self._p.changeVisualShape(objectUniqueId=target.id, linkIndex=-1, rgbaColor=colour)

        return np.array(ob, dtype=np.float32), info

    def calculate_reward_negative_distance(self):
        current_dist_away = self.get_distance_from_desired_stance()

        is_closer = 1 if np.sum(current_dist_away) < np.sum(self.best_dist_to_stance) else 0
        if is_closer: self.best_dist_to_stance = current_dist_away.copy()

        reward = -1 * np.sum(current_dist_away)
        reward += 1000 if self.current_stance == self.desired_stance else 0
        # self.visualise_reward(reward, -6, 0)

        return reward

    def check_reached_stance(self):
        reached = False

        # Check if stance complete
        if self.current_stance == self.desired_stance:
            reached = True

            self.desired_stance_index += 1
            if self.desired_stance_index > len(self.motion_path) - 1: return

            new_stance = self.motion_path[self.desired_stance_index]

            for i, v in enumerate(self.desired_stance):
                self._p.changeVisualShape(objectUniqueId=self.targets[v].id, linkIndex=-1, rgbaColor=[1.0, 0.0, 0.0, 0.75])
            self.desired_stance = new_stance
            for i, v in enumerate(self.desired_stance):
                self._p.changeVisualShape(objectUniqueId=self.targets[v].id, linkIndex=-1, rgbaColor=[0.0, 0.7, 0.1, 0.75])

            # Reset best_dist
            self.best_dist_to_stance = self.get_distance_from_desired_stance()

        return reached

    def update_stance(self):
        self.get_stance_for_effector(0, self.robot.lh_cid)
        self.get_stance_for_effector(1, self.robot.rh_cid)
        self.get_stance_for_effector(2, self.robot.lf_cid)
        self.get_stance_for_effector(3, self.robot.rf_cid)

        if self.render_mode == 'human':
            torso_pos = self.robot.robot_body.current_position()
            torso_pos[1] += 0.15
            torso_pos[2] += 0.35
            self._p.addUserDebugText(text=f"{self.current_stance}", textPosition=torso_pos, textSize=1, lifeTime=1 / 60,
                               textColorRGB=[1.0, 0.0, 1.0])

    def get_stance_for_effector(self, eff_index, eff_cid):
        if eff_cid != -1:
            target_id = self._p.getConstraintInfo(constraintUniqueId=eff_cid)[2]
            for i, target in enumerate(self.targets):
                if target.id == target_id:
                    self.current_stance[eff_index] = i
                    return i
        self.current_stance[eff_index] = -1
        return -1

    def get_distance_from_desired_stance(self):
        dist_away = [float('inf') for _ in range(len(self.robot.effectors))]
        states = self._p.getLinkStates(self.robot.robot, linkIndices=[eff.bodyPartIndex for eff in self.robot.effectors])
        for i, effector in enumerate(self.robot.effectors):
            if self.desired_stance[i] == -1:
                dist_away[i] = 0
                continue

            desired_eff_pos = self._p.getBasePositionAndOrientation(bodyUniqueId=self.targets[self.desired_stance[i]].id)[0]
            current_eff_pos = states[i][0]
            distance = np.abs(np.linalg.norm(np.array(desired_eff_pos) - np.array(current_eff_pos)))
            dist_away[i] = distance
        return dist_away

    def terminate_check(self):
        terminated = True if self.desired_stance_index > len(self.motion_path) - 1 else False
        return terminated

    def truncate_check(self):
        truncated = True if self.steps >= self.max_ep_steps else False
        return truncated

    def _get_obs(self):
        obs = []

        states = self._p.getLinkStates(self.robot.robot, linkIndices=[joint.jointIndex for joint in self.robot.ordered_joints], computeLinkVelocity=1)

        for state in states:
            worldPos, worldOri, localInertialPos, _, _, _, linearVel, angVel = state
            obs += (worldPos + worldOri + localInertialPos + linearVel + angVel)

        eff_positions = [eff.current_position() for eff in self.robot.effectors]
        for i, c_stance in enumerate(self.desired_stance):
            if c_stance == -1:
                obs += [-1, -1, -1, 0]
                continue

            eff_target = self.targets[c_stance]
            dist = np.linalg.norm(np.array(eff_target.pos) - np.array(eff_positions[i]))

            target_obs = eff_target.pos.copy() + [dist]
            obs += target_obs

        obs += self.current_stance
        obs += self.desired_stance
        obs += [1 if self.current_stance[i] == self.desired_stance[i] else 0 for i in range(len(self.current_stance))]
        obs += self.best_dist_to_stance
        obs += [1 if self.is_touching_body(self.floor) else 0]
        obs += [1 if self.is_touching_body(self.wall) else 0]

        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        info = dict()

        success = True if self.current_stance == self.desired_stance else False
        info['is_success'] = success

        return info

    def is_touching_body(self, body, link_indexA=-1):
        contact_points = self._p.getContactPoints(bodyA=self.robot.robot, linkIndexA=link_indexA, bodyB=body)
        return len(contact_points) > 0

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def visualise_reward(self, reward, min, max):
        if self.render_mode != 'human': return
        value = np.clip(reward, min, max)
        normalized_value = (value - min) / (max - min) * (1 - 0) + 0
        colour = [0.0, normalized_value / 1.0, 0.0, 1.0] if reward > 0.0 else [normalized_value / 1.0, 0.0, 0.0, 1.0]
        self._p.changeVisualShape(objectUniqueId=self.robot.robot, linkIndex=-1, rgbaColor=colour)

