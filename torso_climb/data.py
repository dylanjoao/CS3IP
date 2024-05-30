import random

import gymnasium as gym
import pendulum_climb
import torso_climb
from torso_climb.env.torso_climb_env import Reward
import pybullet as p
import time
import pandas as pd
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt
import csv

# MOTION = [[2, 1], [2, 5], [5, 5]]
# env = gym.make('TorsoClimb-v0', render_mode="human", max_ep_steps=1600, reward=Reward.NEGATIVE_DIST, motion_path=MOTION,
#                state_file=None)
# obs, info = env.reset()
#
# state = env.reset()
# done = False
# truncated = False
# score = 0
# step = 0
# pause = False
#
# MODEL_PATH = ["./models/stance1_21.zip", "./models/stance2_25.zip",
#               "./models/stance3_55.zip"]
# MODELS = [PPO.load(MODEL_PATH[i], env=env) for i in range(len(MODEL_PATH))]
# CUR_MODEL = 0
# REWARDS = [0 for i in range(len(MODELS))]
# STEPS = [0 for i in range(len(MODELS))]
# REACHED = [0 for i in range(len(MODELS))]
#
# data = {"stance1_successrate": [], "stance2_successrate": [], "stance3_successrate": [],
#         "stance1_steps": [], "stance2_steps": [], "stance3_steps": [],
#         "stance1_rewards": [], "stance2_rewards": [], "stance3_rewards": []}
#
# episodes = 1
#
# while episodes < 10000:
#     # action = env.action_space.sample()
#
#     if not pause:
#         action, _state = MODELS[CUR_MODEL].predict(obs)
#         obs, reward, done, truncated, info = env.step(action)
#         score += reward
#         step += 1
#
#         REWARDS[CUR_MODEL] += reward
#         STEPS[CUR_MODEL] += 1
#
#         if STEPS[CUR_MODEL] > 200:
#             done = True
#
#     # Reset on backspace
#     keys = p.getKeyboardEvents()
#     #
#     if 114 in keys and keys[114] & p.KEY_WAS_TRIGGERED:
#         # print(f"Score: {score}, Steps {step}")
#         CUR_MODEL = 0
#         REWARDS = [0 for i in range(len(MODELS))]
#         STEPS = [0 for i in range(len(MODELS))]
#         done = False
#         truncated = False
#         pause = False
#         score = 0
#         step = 0
#         env.reset()
#
#     # C
#     # if 99 in keys and keys[99] & p.KEY_WAS_TRIGGERED:
#     # 	CUR_MODEL += 1
#     # 	if CUR_MODEL > len(MODELS)-1:
#     # 		CUR_MODEL = 0
#     # 	print(f"Current model {CUR_MODEL}")
#
#     # Pause on space
#     if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED:
#         pause = not pause
#         print("Paused" if pause else "Unpaused")
#
#     if info["is_success"]:
#         # print(f"Finished stance {CUR_MODEL} with {REWARDS[CUR_MODEL]} reward in {STEPS[CUR_MODEL]} steps")
#         REACHED[CUR_MODEL] = 1
#         CUR_MODEL += 1
#         if CUR_MODEL > len(MODELS) - 1:
#             CUR_MODEL = 0
#
#
#     # Called at the end of the env.
#     if done or truncated:
#         data["stance1_successrate"].append(REACHED[0]/episodes)
#         data["stance1_steps"].append(STEPS[0])
#         data["stance1_rewards"].append(REWARDS[0])
#
#         data["stance2_successrate"].append(REACHED[1])
#         data["stance2_steps"].append(STEPS[1])
#         data["stance2_rewards"].append(REWARDS[1])
#
#         data["stance3_successrate"].append(REACHED[2])
#         data["stance3_steps"].append(STEPS[2])
#         data["stance3_rewards"].append(REWARDS[2])
#
#         CUR_MODEL = 0
#         REWARDS = [0 for i in range(len(MODELS))]
#         STEPS = [0 for i in range(len(MODELS))]
#         REACHED = [0 for i in range(len(MODELS))]
#         env.reset()
#         # print("ENV TERMINATED\n")
#         episodes += 1
#         print(episodes)
#
# env.close()

# df = pd.DataFrame.from_dict(data, orient="columns")
# df.to_csv("./torso_data.csv")

# Load the CSV data into a DataFrame
df = pd.read_csv('torso_data.csv')

# Calculate cumulative success rate of reaching the final stance
cumulative_final_stance_success_rate = df['stance3_successrate'].cumsum() / (df.index + 1) * 100

# Calculate cumulative success rate from each stance to stance
cumulative_stance_to_stance_success_rate = pd.DataFrame()
for col in df.columns[1:4]:
    cumulative_stance_to_stance_success_rate[col] = df[col].cumsum() / (df.index + 1) * 100


# Plot cumulative success rate from each stance to stance
# plt.figure(figsize=(10, 6))
# for i, col in enumerate(cumulative_stance_to_stance_success_rate.columns):
#     plt.plot(df.index, cumulative_stance_to_stance_success_rate[col], label=f'Stance {i} to {i+1}')
# plt.title('Cumulative Success Rate from Each Stance to Stance')
# plt.xlabel('Episodes')
# plt.ylabel('Cumulative Success Rate (%)')
# plt.ylim(0, 105)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Plot time step to each stance
# filtered_df = df[(df[['stance1_steps', 'stance2_steps', 'stance3_steps']] != 0).all(axis=1)]
# downsampled_df = filtered_df.iloc[::100, :]
#
# plt.figure(figsize=(10, 6))
# # for i, col in enumerate(df.columns[4:7]):
#     # plt.plot(df.index, df[f'stance{i+1}_steps'], label=f'Time Step to Stance {col[-1]}')
#     # plt.plot(filtered_df.index, filtered_df[col], label=f'{col} Steps')
# for i, col in enumerate(downsampled_df.columns[4:7]):
#     plt.plot(downsampled_df.index, downsampled_df[col], label=f'Stance {i} to {i+1}')
# plt.title('Time Step to Each Stance')
# plt.xlabel('Episodes')
# plt.ylabel('Time Step')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


filtered_df = df[(df[['stance1_rewards', 'stance2_rewards', 'stance3_rewards']] != 0).all(axis=1)]
downsampled_df = filtered_df.iloc[::25, :]
plt.figure(figsize=(10, 6))
for i, col in enumerate(downsampled_df.columns[7:10]):
    plt.plot(downsampled_df.index, downsampled_df[col], label=f'Stance {i} to {i+1}')
plt.title('Rewards to Each Stance')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()