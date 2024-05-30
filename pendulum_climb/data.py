import os
import random

import pandas as pd

import humanoid_climb
import pendulum_climb
import gymnasium as gym
# import pybullet as p
import time

import humanoid_climb.stances as stances
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt
import csv


env = gym.make('PendulumClimb-v0', render_mode="human")
obs, info = env.reset()

state = env.reset()
done = False
truncated = False
score = 0
step = 0
pause = False

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = ["/models/97fw0eu1/model.zip"]


MODEL = PPO.load(ROOT+MODEL_PATH[0], env=env)

data = {"success_rate": [], "timesteps_till_end": [], "reward_gained": [], "distance_covered": []}

episodes = 1
while episodes < 10000:
    # action = env.action_space.sample()

    if not pause:
        action, _state = MODEL.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)
        score += reward
        step += 1

    # keys = p.getKeyboardEvents()
    #
    # if 114 in keys and keys[114] & p.KEY_WAS_TRIGGERED:
    #     # print(f"Score: {score}, Steps {step}")
    #     done = False
    #     truncated = False
    #     pause = False
    #     score = 0
    #     step = 0
    #     env.reset()
    #
    # # Pause on space
    # if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED:
    #     pause = not pause
    #     print("Paused" if pause else "Unpaused")
    #     print(info["distance"])

    if done or truncated:
        reached = 1 if info["distance"] < 2.1 else 0
        data["success_rate"].append(reached)
        data["timesteps_till_end"].append(step)
        data["reward_gained"].append(score)
        data["distance_covered"].append(info["distance"])

        env.reset()
        episodes += 1
        step = 0
        score = 0
        print(episodes)

    # time.sleep(1/30)


env.close()
#
# df = pd.DataFrame.from_dict(data, orient="columns")
# df.to_csv("./pendulum_data.csv")

# Load the CSV file into a DataFrame
# df = pd.read_csv('pendulum_data.csv')

# # Extract data for each metric
# episodes = range(1, len(df) + 1)
# success_rate = df['success_rate']
# timesteps_till_end = df['timesteps_till_end']
# reward_gained = df['reward_gained']
# distance_covered = df['distance_covered']
# cumulative_success_rate = df['success_rate'].cumsum() / (df.index + 1) * 100
#
#
# # Create subplots
# fig, axs = plt.subplots(4, 1, figsize=(10, 12))
#
# # Plot success rate
# axs[0].plot(episodes, cumulative_success_rate, color='blue')
# axs[0].set_title('Success Rate')
# axs[0].set_xlabel('Episodes')
# axs[0].set_ylabel('Success Rate')
# axs[0].set_ylim([0, 105])
# axs[0].grid(True)
#
# # Plot timesteps till end
# axs[1].plot(episodes, timesteps_till_end, color='orange')
# axs[1].set_title('Timesteps till End')
# axs[1].set_xlabel('Episodes')
# axs[1].set_ylabel('Timesteps')
# axs[1].grid(True)
#
# # Plot reward gained
# axs[2].plot(episodes, reward_gained, color='green')
# axs[2].set_title('Reward Gained')
# axs[2].set_xlabel('Episodes')
# axs[2].set_ylabel('Reward')
# axs[2].grid(True)
# axs[2].ticklabel_format(useOffset=False)
#
# # Plot distance covered
# axs[3].plot(episodes, distance_covered, color='red')
# axs[3].set_title('Distance Covered')
# axs[3].set_xlabel('Episodes')
# axs[3].set_ylabel('Distance')
# axs[3].grid(True)
#
# # Adjust layout
# plt.tight_layout()
#
#
# # Show the plot
# plt.show()

# Extract data for each metric
# episodes = range(1, len(df) + 1)
# success_rate = df['success_rate']
# timesteps_till_end = df['timesteps_till_end']
# reward_gained = df['reward_gained']
# distance_covered = df['distance_covered']
# cumulative_success_rate = df['success_rate'].cumsum() / (df.index + 1) * 100
#
# # Plot Success Rate
# plt.figure(figsize=(10, 6))
# plt.plot(episodes, cumulative_success_rate, color='blue')
# plt.title('Success Rate')
# plt.xlabel('Episodes')
# plt.ylabel('Success Rate (%)')
# plt.ylim(0, 105)  # Set y-axis limit from 0 to 105%
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # Plot Timesteps till End
# plt.figure(figsize=(10, 6))
# plt.plot(episodes, timesteps_till_end, color='orange')
# plt.title('Timesteps till End')
# plt.xlabel('Episodes')
# plt.ylabel('Timesteps')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # Plot Reward Gained
# plt.figure(figsize=(10, 6))
# plt.plot(episodes, reward_gained, color='green')
# plt.ticklabel_format(useOffset=False)
# plt.title('Reward Gained')
# plt.xlabel('Episodes')
# plt.ylabel('Reward')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # Plot Distance Covered
# plt.figure(figsize=(10, 6))
# plt.plot(episodes, distance_covered, color='red')
# plt.title('Distance Covered')
# plt.xlabel('Episodes')
# plt.ylabel('Distance')
# plt.grid(True)
# plt.tight_layout()
# plt.show()