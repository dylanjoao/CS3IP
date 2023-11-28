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
        # +vel, -vel, grasp, release
        self.action_space = gym.spaces.Discrete(8)

        self.observation_space = gym.spaces.Dict(
            {
                "agent_position": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,), dtype=np.float32),
                "agent_angle": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,), dtype=np.float32),
                "agent_velocity": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,), dtype=np.float32),
                "agent_holds": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.integer),
                "target_position": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,), dtype=np.float32)
            }
        )

        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.GUI)
        self.pendulum = None
        self.pendulum_pos = []
        self.goal = None
        self.initial_dist = None
        self.targets = []
        self._max_episode_steps = 1000
        self._elapsed_steps = 0

        # configure pybullet GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=90, cameraPitch=0,
                                     cameraTargetPosition=[0, 0, 5])

        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1 / 30, self.client)

    def _get_obs(self):
        pen_ob = self.pendulum.get_observation()

        return {"agent_position": pen_ob["pos"],
                "agent_angle": pen_ob["ang"],
                "agent_velocity": pen_ob["vel"],
                "agent_holds": pen_ob["hold"],
                "target_position": self.goal}

    def _get_info(self):
        return {"distance:": np.linalg.norm(np.array(self.pendulum_pos) - np.array(self.goal))}

    def step(self, action):
        # Feed action to the pendulum and get observation of pendulum's state
        self.pendulum.apply_action(action)

        p.stepSimulation()

        # Gather information about the env
        ob = self._get_obs()
        info = self._get_info()

        # Update values
        agent_position = ob["agent_position"]
        dist_to_goal = np.linalg.norm(np.array(agent_position) - np.array(self.goal))
        self.pendulum_pos = agent_position

        # Quadratic reward
        reward = (self.initial_dist / dist_to_goal) ** 2

        # Check termination conditions
        terminated = False
        truncated = False
        if dist_to_goal < 0.05:
            terminated = True
            reward = 50
        elif agent_position[2] < 0.8 or agent_position[2] > 50:
            terminated = True

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
        self._elapsed_steps += 1

        return ob, reward, terminated, truncated, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        # Reload the plane and car
        plane = p.loadURDF("plane.urdf")
        self.pendulum = Pendulum(self.client, [0, 0, 2.5])
        self.targets.clear()

        # Targets equally apart
        dist = 1.0
        for i in range(10):
            target = Target(self.client, [0, 0, i + 1 * 2 + dist])
            self.targets.append(target)

        initial_constraint = p.createConstraint(parentBodyUniqueId=self.pendulum.id,
                                                parentLinkIndex=0,
                                                childBodyUniqueId=self.targets[0].id,
                                                childLinkIndex=-1,
                                                jointType=p.JOINT_POINT2POINT,
                                                jointAxis=[0, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])
        self.pendulum.top_held = self.targets[0]
        self.pendulum.targets = self.targets
        self.pendulum.top_held = initial_constraint
        self.targets[0].constraint = initial_constraint

        goal_pos, _ = p.getBasePositionAndOrientation(self.targets[-1].id, self.client)
        self.goal = goal_pos

        # Get observation to return
        ob = self._get_obs()
        agent_position = ob["agent_position"]

        self.pendulum_pos = agent_position
        self.initial_dist = np.linalg.norm(np.array(agent_position) - np.array(goal_pos))
        self._elapsed_steps = 0

        info = self._get_info()

        return ob, info

    def render(self, mode='human'):
        pass
        # self.client = p.connect(p.DIRECT)

    def close(self):
        p.disconnect(self.client)
