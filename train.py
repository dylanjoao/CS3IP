import time
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
import pybullet as p
import stable_baselines3 as sb
import os
import argparse

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, VecFrameStack
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback

from torso_climb.env.torso_climb_env import Reward
from typing import List

import wandb
from wandb.integration.sb3 import WandbCallback

import pendulum_climb
import torso_climb

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


class CustomCallback(BaseCallback):
	def __init__(self, verbose: int = 0):
		super().__init__(verbose)
		self.rollout_count = 0

	def _on_step(self) -> bool:
		return True

	def _on_rollout_end(self) -> None:
		self.rollout_count += 1
		_steps_till_success = []
		_success = []
		_best_dist_lh = []
		_best_dist_rh = []
		_final_dist_lh = []
		_final_dist_rh = []

		for entry in self.model.ep_info_buffer:
			if entry['steps_till_success'] > 0: _steps_till_success.append(entry['steps_till_success'])
			_success.append(entry['is_success'])
			_best_dist_lh.append(entry['best_dist_lh'])
			_best_dist_rh.append(entry['best_dist_rh'])
			_final_dist_lh.append(entry['final_dist_lh'])
			_final_dist_rh.append(entry['final_dist_rh'])

		mean_steps_till_success = np.mean(_steps_till_success) if len(_steps_till_success) > 0 else None
		mean_best_dist_lh = np.mean(_best_dist_lh) if len(_best_dist_lh) > 0 else None
		mean_best_dist_rh = np.mean(_best_dist_rh) if len(_best_dist_rh) > 0 else None
		mean_final_dist_lh = np.mean(_final_dist_lh) if len(_final_dist_lh) > 0 else None
		mean_final_dist_rh = np.mean(_final_dist_rh) if len(_final_dist_rh) > 0 else None
		success_rate = np.mean(_success) if len(_final_dist_rh) > 0 else None

		self.logger.record("climb/success_rate", success_rate)
		self.logger.record("climb/mean_steps_success", mean_steps_till_success)
		self.logger.record("climb/mean_best_dist_lh", mean_best_dist_lh)
		self.logger.record("climb/mean_best_dist_rh", mean_best_dist_rh)
		self.logger.record("climb/mean_final_dist_lh", mean_final_dist_lh)
		self.logger.record("climb/mean_final_dist_rh", mean_final_dist_rh)
		self.logger.record("climb/rollout_count", self.rollout_count)


def make_env(env_id: str, rank: int, seed: int = 0, max_steps: int = 600, reward: Reward = Reward.EQ1, motion_path: List[List[int]] = [[2, 1]], state_file: str = None):
	def _init():
		env = gym.make(env_id, render_mode=None, max_ep_steps=max_steps, reward=reward, motion_path=motion_path, state_file=state_file)
		m_env = Monitor(env, info_keywords=('is_success', 'steps_till_success', 'best_dist_lh', 'best_dist_rh', 'final_dist_lh', 'final_dist_rh'))
		m_env.reset(seed=seed + rank)
		return m_env

	set_random_seed(seed)
	return _init


def train(env_name, sb3_algo, workers, path_to_model=None):
	config = {
		"policy_type": "MlpPolicy",
		"total_timesteps": 50000000,
		"env_name": env_name,
	}
	run = wandb.init(
		project="torsoclimb_climb_take3",
		config=config,
		sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
		monitor_gym=False,  # auto-upload the videos of agents playing the game
		save_code=False,  # optional
	)

	max_ep_steps = 600
	reward_type = Reward.NEGATIVE_DIST
	motion_path = [[6, 5]]
	state_file_path = "./torso_climb/states/stance2_25.npz"
	vec_env = SubprocVecEnv([make_env(env_name, i, max_steps=max_ep_steps, reward=reward_type, motion_path=motion_path, state_file=state_file_path) for i in range(workers)], start_method="spawn")

	model = None
	save_path = f"{model_dir}/{run.id}"

	eval_callback = EvalCallback(vec_env, best_model_save_path=f"{save_path}/models/", log_path=f"{save_path}/logs/", eval_freq=500, deterministic=True, render=False)
	cust_callback = CustomCallback()

	if sb3_algo == 'PPO':
		if path_to_model is None:
			model = sb.PPO('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)
		else:
			model = sb.PPO.load(path_to_model, env=vec_env)
	elif sb3_algo == 'SAC':
		if path_to_model is None:
			model = sb.SAC('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)
		else:
			model = sb.SAC.load(path_to_model, env=vec_env)
	else:
		print('Algorithm not found')
		return

	model.learn(
		total_timesteps=config["total_timesteps"],
		progress_bar=True,
		callback=[WandbCallback(
			gradient_save_freq=5000,
			model_save_freq=5000,
			model_save_path=save_path,
			verbose=2,
		), eval_callback, cust_callback],
	)
	run.finish()


def test(env, sb3_algo, path_to_model):
	if sb3_algo == 'SAC':
		model = sb.SAC.load(path_to_model, env=env)
	elif sb3_algo == 'TD3':
		model = sb.TD3.load(path_to_model, env=env)
	elif sb3_algo == 'A2C':
		model = sb.A2C.load(path_to_model, env=env)
	elif sb3_algo == 'DQN':
		model = sb.DQN.load(path_to_model, env=env)
	elif sb3_algo == 'PPO':
		model = sb.PPO.load(path_to_model, env=env)
	else:
		print('Algorithm not found')
		return

	vec_env = model.get_env()
	obs = vec_env.reset()
	score = 0
	step = 0

	while True:
		action, _state = model.predict(obs, deterministic=True)
		obs, reward, done, info = vec_env.step(action)
		score += reward
		step += 1

		# env.reset() auto called on vec_env?
		if done:
			print(f"Episode Over, Score: {score}, Steps {step}")
			score = 0
			step = 0

		# Reset on backspace
		keys = p.getKeyboardEvents()
		if 114 in keys and keys[114] & p.KEY_WAS_TRIGGERED:
			score = 0
			step = 0
			env.reset()

	env.close()


if __name__ == '__main__':

	# Parse command line inputs
	parser = argparse.ArgumentParser(description='Train or test model.')
	parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
	parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
	parser.add_argument('-w', '--workers', type=int)
	parser.add_argument('-t', '--train', action='store_true')
	parser.add_argument('-f', '--file', required=False, default=None)
	parser.add_argument('-s', '--test', metavar='path_to_model')
	args = parser.parse_args()

	if args.train:
		if args.file is None:
			print(f'<< Training from scratch! >>')
			train(args.gymenv, args.sb3_algo, args.workers)
		elif os.path.isfile(args.file):
			print(f'<< Continuing {args.file} >>')
			train(args.gymenv, args.sb3_algo, args.workers, args.file)

	if args.test:
		if os.path.isfile(args.test):
			max_steps = 600
			reward = Reward.NEGATIVE_DIST
			stance = [[6, 5]]
			state_file = "./torso_climb/states/stance2_25.npz"
			env = gym.make(args.gymenv, render_mode='human', reward=reward, motion_path=stance, state_file=state_file)
			test(env, args.sb3_algo, path_to_model=args.test)
		else:
			print(f'{args.test} not found.')
