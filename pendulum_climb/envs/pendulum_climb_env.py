import os
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
        self.client = p.connect(p.GUI)

        # +/-velX joint1, +/-velY joint1
        # +/-velX joint2, +/-velY joint2
        # grasp
        # release
        self.action_space = gym.spaces.Discrete(10)

        # Holds[2], DistanceAwayFromNextHold[2], Velocity[3]
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,), dtype=np.float32)

        self.pendulum = None
        self.targets = []
        self.next_target = None

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _get_obs(self):
        holds = [0.0 if x is None else 1.0 for x in self.pendulum.constraints]
        dist = [0.0, 0.0]
        velocity, _ = p.getBaseVelocity(self.pendulum.id, self.client)

        next_target_pos, _ = p.getBasePositionAndOrientation(self.next_target.id, self.client)
        joint_one_pos, _, _, _, _, _ = p.getLinkState(bodyUniqueId=self.pendulum.id, linkIndex=0,
                                                      physicsClientId=self.client)
        joint_two_pos, _, _, _, _, _ = p.getLinkState(bodyUniqueId=self.pendulum.id, linkIndex=1,
                                                      physicsClientId=self.client)
        dist[0] = np.linalg.norm(np.array(joint_one_pos) - np.array(next_target_pos))
        dist[1] = np.linalg.norm(np.array(joint_two_pos) - np.array(next_target_pos))

        return np.concatenate((holds, dist, velocity), dtype=np.float32)

    def _get_info(self):
        goal_target_pos, _ = p.getBasePositionAndOrientation(self.targets[-1].id, self.client)
        pendulum_pos, _ = p.getBasePositionAndOrientation(self.pendulum.id, self.client)
        distance_from_goal = np.linalg.norm(np.array(pendulum_pos) - np.array(goal_target_pos))
        return {"distance": distance_from_goal}

    def step(self, action):
        p.stepSimulation(self.client)




        obs, info = self._get_obs(), self._get_info()

        return obs, 0.0, False, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        plane = p.loadURDF("plane.urdf")
        self.pendulum = Pendulum(self.client, [0, 0, 2.0])
        self.targets.clear()

        dist = 9.0
        for i in range(10):
            target = Target(self.client, [0, 0, 3+i * 2.0])
            self.targets.append(target)

        self.next_target = self.targets[1]

        initial_constraint = p.createConstraint(parentBodyUniqueId=self.pendulum.id,
                                                parentLinkIndex=0,
                                                childBodyUniqueId=self.targets[0].id,
                                                childLinkIndex=-1,
                                                jointType=p.JOINT_POINT2POINT,
                                                jointAxis=[0, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])
        self.pendulum.constraints[0] = initial_constraint

        obs, info = self._get_obs(), self._get_info()

        return obs, info

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.client)
