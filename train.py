import time
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import pybullet as p
import stable_baselines3 as sb
import os
import argparse

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

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
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def train(env_name, sb3_algo, workers):
    vec_env = SubprocVecEnv([make_env(env_name, i) for i in range(workers)])

    if sb3_algo == 'SAC':
        model = sb.SAC('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif sb3_algo == 'TD3':
        model = sb.TD3('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif sb3_algo == 'A2C':
        model = sb.A2C('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif sb3_algo == 'DQN':
        model = sb.DQN('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif sb3_algo == 'PPO':
        model = sb.PPO('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)
    else:
        print('Algorithm not found')
        return

    TIMESTEPS = 25000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS * iters}")



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

        if done:
            print(f"Episode Over, Score: {score}, Steps {step}")

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
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    if args.train:
        train(args.gymenv, args.sb3_algo, args.workers)

    if args.test:
        if os.path.isfile(args.test):
            env = gym.make(args.gymenv, render_mode='human')
            test(env, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')
