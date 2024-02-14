import time
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import pybullet as p
import stable_baselines3 as sb
import os
import argparse

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

import wandb
from wandb.integration.sb3 import WandbCallback

import pendulum_climb
import torso_climb

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """

    def _init():
        env = gym.make(env_id, max_ep_steps=1000)
        m_env = Monitor(env, info_keywords=('steps_till_first_hold_reached_lh', 'steps_till_first_hold_reached_rh'))
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
        project="sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    vec_env = SubprocVecEnv([make_env(env_name, i) for i in range(workers)], start_method="spawn")

    model = None

    if sb3_algo == 'PPO':
        if path_to_model is None: model = sb.PPO('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)
        else: model = sb.PPO.load(path_to_model, env=vec_env)
    elif sb3_algo == 'SAC':
        if path_to_model is None: model = sb.SAC('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)
        else: model = sb.SAC.load(path_to_model, env=vec_env)
    elif sb3_algo == 'TD3':
        if path_to_model is None: model = sb.TD3('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif sb3_algo == 'A2C':
        model = sb.A2C('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif sb3_algo == 'DQN':
        model = sb.DQN('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)
    else:
        print('Algorithm not found')
        return

    model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        callback=WandbCallback(
            gradient_save_freq=5000,
            model_save_freq=5000,
            model_save_path=f"{model_dir}/{run.id}",
            verbose=2,
        ),
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
        if 65305 in keys and keys[65305] & p.KEY_WAS_TRIGGERED:
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
            env = gym.make(args.gymenv, render_mode='human')
            test(env, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')
