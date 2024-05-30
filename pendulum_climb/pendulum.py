import os
import random

import gymnasium as gym
import pendulum_climb
import torso_climb
import humanoid_climb
from torso_climb.env.torso_climb_env import Reward
import pybullet as p
import time
from stable_baselines3 import PPO, SAC
import humanoid_climb.stances as stances

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, VecFrameStack
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
import stable_baselines3 as sb
import wandb
from wandb.integration.sb3 import WandbCallback


# env = gym.make('TorsoClimb-v0', render_mode='human', max_ep_steps=600, reward=Reward.NEGATIVE_DIST, motion_path=MOTION, state_file=STATEFILE)

def make_env(env_id: str, rank: int, seed: int = 0) -> gym.Env:
    def _init():
        env = gym.make(env_id)
        m_env = Monitor(env)
        m_env.reset(seed=seed + rank)
        return m_env

    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    env_name = "PendulumClimb-v0"


    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 50000000,
        "env_name": env_name,\
    }
    run = wandb.init(
        project="PendulumClimb",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
        # id="6sbyjyfr"
    )

    vec_env = SubprocVecEnv([make_env(env_name, i) for i in range(10)],
                            start_method="spawn")

    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = None
    save_path = f"{model_dir}/{run.id}"

    eval_callback = EvalCallback(vec_env, best_model_save_path=f"{save_path}/models/", log_path=f"{save_path}/logs/",
                                 eval_freq=500, deterministic=True, render=False)

    model = sb.PPO('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)

    model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        callback=[WandbCallback(
            gradient_save_freq=5000,
            model_save_freq=5000,
            model_save_path=save_path,
            verbose=2,
        ), eval_callback],
    )
    run.finish()

"""
ob, info = env.reset(seed=42)

state = env.reset()
done = False
truncated = False
score = 0
step = 0
pause = False
hold = True

action = 1

while True:

    if not pause:
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        step += 1

    # Reset on backspace
    keys = p.getKeyboardEvents()

    # rarrow
    if 65296 in keys and keys[65296] & p.KEY_WAS_TRIGGERED:
        pass

    # r
    if 114 in keys and keys[114] & p.KEY_WAS_TRIGGERED:
        print(f"Score: {score}, Steps {step}")
        done = False
        truncated = False
        pause = False
        score = 0
        step = 0
        env.reset()

    # Pause on space
    if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED:
        pause = not pause
        print("Paused" if pause else "Unpaused")

    if done or truncated:
        pause = True

    time.sleep(0.1)

env.close()
"""
