import os
from pprint import pprint
from time import sleep

import gymnasium as gym
import numpy as np
import math
import pybullet as p
import pybullet_data

from pendulum_climb.assets.pendulum import Pendulum
from pendulum_climb.assets.target import Target


class PendulumClimbEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.client = p.connect(p.GUI_SERVER)

        # +/-velX joint1,
        # +/-velX joint2,
        # grasp
        # release
        self.action_space = gym.spaces.Discrete(8)

        # Holds[2], DistanceAwayFromNextHold[2], Velocity[3]
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,), dtype=np.float32)

        self.pendulum = None
        self.targets = []
        self.target_order = []
        self.next_target = None
        self.best_distance = float('inf')
        self.prev_dist = 0.0

        self.steps = 0
        self.max_steps = 1500

        # configure pybullet GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[0, 0, 5])

    def _get_obs(self):
        holds = [0.0 if x is None else 1.0 for x in self.pendulum.constraints]
        dist = [0.0, 0.0]
        velocity, _ = p.getBaseVelocity(self.pendulum.id, physicsClientId=self.client)

        next_target_pos, _ = p.getBasePositionAndOrientation(self.next_target.id, physicsClientId=self.client)
        joint_one_pos, _, _, _, _, _ = p.getLinkState(bodyUniqueId=self.pendulum.id, linkIndex=0,
                                                      physicsClientId=self.client)
        joint_two_pos, _, _, _, _, _ = p.getLinkState(bodyUniqueId=self.pendulum.id, linkIndex=1,
                                                      physicsClientId=self.client)
        dist[0] = np.linalg.norm(np.array(joint_one_pos) - np.array(next_target_pos))
        dist[1] = np.linalg.norm(np.array(joint_two_pos) - np.array(next_target_pos))

        return np.concatenate((holds, dist, velocity), dtype=np.float32)

    def _get_info(self):
        goal_target_pos, _ = p.getBasePositionAndOrientation(self.targets[-1].id, physicsClientId=self.client)
        next_target_pos, _ = p.getBasePositionAndOrientation(self.targets[0].id, physicsClientId=self.client)
        pendulum_pos, _ = p.getBasePositionAndOrientation(self.pendulum.id, physicsClientId=self.client)
        distance_from_goal = np.linalg.norm(np.array(pendulum_pos) - np.array(goal_target_pos))
        distance_from_next_target = np.linalg.norm(np.array(pendulum_pos) - np.array(next_target_pos))
        return {"overall_distance": distance_from_goal, "next_distance": distance_from_next_target}

    def step(self, action):
        p.stepSimulation(physicsClientId=self.client)

        reward, terminated, truncated = 0.0, False, False

        self.pendulum.apply_action(action)

        # Check if attached to next target
        for constraint in self.pendulum.constraints:
            if constraint is None: continue
            target = p.getConstraintInfo(constraintUniqueId=constraint, physicsClientId=self.client)[2]
            if target != self.next_target.id: continue
            self.target_order.pop(0)

            # If last target
            if len(self.target_order) == 0:
                terminated = True
                reward += 100.0
            else:
                self.next_target = self.target_order[0]
                reward += 10.0
                self.best_distance = float('inf')

        # Main Reward
        pendulum_pos, _ = p.getBasePositionAndOrientation(self.pendulum.id, physicsClientId=self.client)
        next_target_pos, _ = p.getBasePositionAndOrientation(self.next_target.id, physicsClientId=self.client)
        distance_away = np.linalg.norm(np.array(pendulum_pos[2]) - np.array(next_target_pos[2]))
        if distance_away <= self.best_distance:
            self.best_distance = distance_away
            reward += 1.0 * (1-(self.steps/self.max_steps))
        else:
            reward -= 1.0
            # terminated = True
        self.prev_dist = distance_away

        # Check termination condition
        pendulum_pos, _ = p.getBasePositionAndOrientation(self.pendulum.id, physicsClientId=self.client)
        if pendulum_pos[2] < 1.0:
            terminated = True

        self.steps += 1
        if self.steps > self.max_steps:
            truncated = True

        obs, info = self._get_obs(), self._get_info()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -10)

        plane = p.loadURDF("plane.urdf")
        self.pendulum = Pendulum(self.client, [0, 0, 2.0])
        self.targets.clear()
        self.target_order.clear()
        self.steps = 0
        self.prev_dist = 0.0
        self.best_distance = float('inf')

        dist = 9.0
        for i in range(10):
            target = Target(self.client, [0, 0, 3 + i * 2.0])
            self.targets.append(target)

        self.target_order = self.targets.copy()[1:-1]
        self.pendulum.targets = self.targets.copy()
        self.next_target = self.targets[1]

        initial_constraint = p.createConstraint(parentBodyUniqueId=self.pendulum.id,
                                                parentLinkIndex=0,
                                                childBodyUniqueId=self.targets[0].id,
                                                childLinkIndex=-1,
                                                jointType=p.JOINT_POINT2POINT,
                                                jointAxis=[0, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0],
                                                physicsClientId=self.client)
        self.pendulum.constraints[0] = initial_constraint

        obs, info = self._get_obs(), self._get_info()

        return obs, info

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.client)
