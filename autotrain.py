import os

import wandb
import humanoid_climb
import stable_baselines3 as sb
import gymnasium as gym

from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from humanoid_climb.env.climbing_config import ClimbingConfig
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# Create env

# Once training terminated
#   if 0% success rate we have a problem
# Generate final pose states for the stance

# Create env

def make_env(env_id: str, rank: int, config, seed: int = 0, max_steps: int = 1000) -> gym.Env:
    def _init():
        env = gym.make(env_id, render_mode=None, max_ep_steps=max_steps, config=config)
        m_env = Monitor(env)
        m_env.reset(seed=seed + rank)
        return m_env

    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    model_dir = "models"
    log_dir = "logs"
    states_dir = "states"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(states_dir, exist_ok=True)


    config = ClimbingConfig('./config.json')
    stance_path = list(config.stance_path.copy().items())
    stance_transitions = [stance_path[i:i+2] for i in range(len(stance_path)-1)]
    current_stance_transition = 0

    wandb_config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 8000000,
        "env_name": "HumanoidClimb-v0",
    }

    for i in range(1):
        run = wandb.init(
            project="HumanoidClimb-3",
            config=wandb_config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=False,  # auto-upload the videos of agents playing the game
            save_code=False,  # optional
            # id="6sbyjyfr"
        )

        config.stance_path = dict(stance_transitions[i])

        vec_env = SubprocVecEnv([make_env("HumanoidClimb-v0", i, max_steps=600, config=config) for i in range(4)],
                                start_method="spawn")

        save_path = f"{model_dir}/{run.id}"
        eval_callback = EvalCallback(vec_env, best_model_save_path=f"{save_path}/models/", log_path=f"{save_path}/logs/",
                                     eval_freq=500, deterministic=True, render=False)
        model = sb.PPO('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)

        # Train for max 8m steps
        model.learn(
            total_timesteps=wandb_config["total_timesteps"],
            progress_bar=True,
            callback=[WandbCallback(
                gradient_save_freq=5000,
                model_save_freq=5000,
                model_save_path=save_path,
                verbose=2,
            ), eval_callback],
        )
        run.finish()

        # Create states for finished stance



