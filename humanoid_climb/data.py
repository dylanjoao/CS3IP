import os
import random

import pandas as pd
from matplotlib import pyplot as plt

import humanoid_climb
import gymnasium as gym
import pybullet as p
import time

import humanoid_climb.stances as stances
from stable_baselines3 import PPO, SAC

# stances.set_root_path("./")
# STANCES = [stances.STANCE_1, stances.STANCE_2, stances.STANCE_3, stances.STANCE_4, stances.STANCE_5, stances.STANCE_6,
#            stances.STANCE_7, stances.STANCE_8, stances.STANCE_9, stances.STANCE_10, stances.STANCE_11_3, stances.STANCE_12,
#            stances.STANCE_13_2, stances.STANCE_14]
#
# MOTION = [s.stance for s in STANCES]
# EXCLUDE = [s.exclude_targets for s in STANCES]
# O_ACTION = [s.action_override for s in STANCES]
#
# env = gym.make('HumanoidClimb-v0', render_mode=None, max_ep_steps=50000, motion_path=MOTION, state_file=None, motion_exclude_targets=EXCLUDE)
# obs, info = env.reset()
#
# state = env.reset()
# done = False
# truncated = False
# score = 0
# step = 0
# pause = False
#
# STANCE_TOLERANCE = 700
# ROOT = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = ["/models/1_10_9_n_n.zip",
#               "/models/2_10_9_2_n.zip",
#               "/models/3_10_9_2_1.zip",
#               "/models/4_10_13_2_1.zip",
#               "/models/5_10_13_2_5.zip",
#               "/models/6_13_13_n_5.zip",
#               "/models/7_13_13_6_5.zip",
#               "/models/8_14_13_6_5.zip",
#               "/models/9_14_17_6_5.zip",
#               "/models/10_14_17_n_9.zip",
#               "/models/11_14_17_10_9.zip",
#               "/models/12_18_17_10_9.zip",
#               "/models/13_18_20_10_9.zip",
#               "/models/14_20_20_10_9.zip"]
#
#
# MODELS = [PPO.load(ROOT+MODEL_PATH[i], env=env) for i in range(len(MODEL_PATH))]
# CUR_MODEL = 0
# REWARDS = [0 for i in range(len(MODELS))]
# STEPS = [0 for i in range(len(MODELS))]
# REACHED = [0 for i in range(len(MODELS))]
#
# last_completed_stance = None
# climb_attempts = 0
# successful_attempts = 0
#
# data = {
#     "stance1_successrate": [],
#     "stance2_successrate": [],
#     "stance3_successrate": [],
#     "stance4_successrate": [],
#     "stance5_successrate": [],
#     "stance6_successrate": [],
#     "stance7_successrate": [],
#     "stance8_successrate": [],
#     "stance9_successrate": [],
#     "stance10_successrate": [],
#     "stance11_successrate": [],
#     "stance12_successrate": [],
#     "stance13_successrate": [],
#     "stance14_successrate": [],
#
#     "stance1_steps": [],
#     "stance2_steps": [],
#     "stance3_steps": [],
#     "stance4_steps": [],
#     "stance5_steps": [],
#     "stance6_steps": [],
#     "stance7_steps": [],
#     "stance8_steps": [],
#     "stance9_steps": [],
#     "stance10_steps": [],
#     "stance11_steps": [],
#     "stance12_steps": [],
#     "stance13_steps": [],
#     "stance14_steps": [],
#
#     "stance1_rewards": [],
#     "stance2_rewards": [],
#     "stance3_rewards": [],
#     "stance4_rewards": [],
#     "stance5_rewards": [],
#     "stance6_rewards": [],
#     "stance7_rewards": [],
#     "stance8_rewards": [],
#     "stance9_rewards": [],
#     "stance10_rewards": [],
#     "stance11_rewards": [],
#     "stance12_rewards": [],
#     "stance13_rewards": [],
#     "stance14_rewards": [],
# }
#
# episodes = 1
# while episodes < 1000:
#     # action = env.action_space.sample()
#
#     if not pause:
#         action, _state = MODELS[CUR_MODEL].predict(obs, deterministic=True)
#
#         for i in range(4):
#             if O_ACTION[CUR_MODEL][i] != -1:
#                 action[17+i] = O_ACTION[CUR_MODEL][i]
#
#         obs, reward, done, truncated, info = env.step(action)
#         score += reward
#         step += 1
#
#         REWARDS[CUR_MODEL] += reward
#         STEPS[CUR_MODEL] += 1
#
#
#     # Reset on backspace
#     keys = p.getKeyboardEvents()
#     #
#     if 114 in keys and keys[114] & p.KEY_WAS_TRIGGERED:
#         # print(f"Score: {score}, Steps {step}")
#         CUR_MODEL = 0
#         REWARDS = [0 for i in range(len(MODELS))]
#         STEPS = [0 for i in range(len(MODELS))]
#         REACHED = [0 for i in range(len(MODELS))]
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
#     if STEPS[CUR_MODEL] > STANCE_TOLERANCE:
#         truncated = True
#         STEPS[CUR_MODEL] = 0
#         REWARDS[CUR_MODEL] = 0
#
#
#     if info["is_success"]:
#         REACHED[CUR_MODEL] = 1
#         CUR_MODEL += 1
#         if CUR_MODEL > len(MODELS) - 1:
#             CUR_MODEL = 0
#
#     if done or truncated:
#
#         for i in range(len(MODELS)):
#             data[f"stance{i+1}_successrate"].append(REACHED[i])
#             data[f"stance{i+1}_steps"].append(STEPS[i])
#             data[f"stance{i+1}_rewards"].append(REWARDS[i])
#
#         CUR_MODEL = 0
#         REWARDS = [0 for i in range(len(MODELS))]
#         STEPS = [0 for i in range(len(MODELS))]
#         REACHED = [0 for i in range(len(MODELS))]
#         env.reset()
#
#         print(episodes)
#         episodes += 1
#
#
# env.close()
# df = pd.DataFrame.from_dict(data, orient="columns")
# df.to_csv("./humanoid_data_1.csv")


# Load the CSV data into a DataFrame
df = pd.read_csv('humanoid_data_1.csv')

# Calculate cumulative success rate of reaching the final stance
cumulative_final_stance_success_rate = df['stance14_successrate'].cumsum() / (df.index + 1) * 100

# Calculate cumulative success rate from each stance to stance
cumulative_stance_to_stance_success_rate = pd.DataFrame()
for col in df.columns[1:15]:
    cumulative_stance_to_stance_success_rate[col] = df[col].cumsum() / (df.index + 1) * 100

# # Plot cumulative success rate from each stance to stance
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
# downsampled_df = filtered_df.iloc[::25, :]
#
# plt.figure(figsize=(10, 6))
# # for i, col in enumerate(df.columns[4:7]):
#     # plt.plot(df.index, df[f'stance{i+1}_steps'], label=f'Time Step to Stance {col[-1]}')
#     # plt.plot(filtered_df.index, filtered_df[col], label=f'{col} Steps')
# for i, col in enumerate(downsampled_df.columns[15:29]):
#     plt.plot(downsampled_df.index, downsampled_df[col], label=f'Stance {i} to {i+1}')
# plt.title('Time Step to Each Stance')
# plt.xlabel('Episodes')
# plt.ylabel('Time Step')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

filtered_df = df[(df[['stance1_rewards', 'stance2_rewards', 'stance3_rewards',
                      'stance4_rewards',
                      'stance5_rewards',
                      'stance6_rewards',
                      'stance7_rewards',
                      'stance8_rewards',
                      'stance9_rewards',
                      'stance10_rewards',
                      'stance11_rewards',
                      'stance12_rewards',
                      'stance13_rewards',
                      'stance14_rewards',]] != 0).all(axis=1)]
downsampled_df = filtered_df.iloc[::2, :]
plt.figure(figsize=(10, 6))
for i, col in enumerate(downsampled_df.columns[29:43]):
    plt.plot(downsampled_df.index, downsampled_df[col], label=f'Stance {i} to {i+1}')
plt.title('Rewards to Each Stance')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()