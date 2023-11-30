import time
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import stable_baselines3 as sb
import os
import argparse
import pendulum_climb

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def train(env, sb3_algo):
    if sb3_algo == 'SAC':
        model = sb.SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif sb3_algo == 'TD3':
        model = sb.TD3('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif sb3_algo == 'A2C':
        model = sb.A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif sb3_algo == 'DQN':
        model = sb.DQN('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif sb3_algo == 'PPO':
        model = sb.PPO('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)        
    else:
        print('Algorithm not found')
        return

    TIMESTEPS = 25000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS * iters}")


def cont_train(env, sb3_algo, path_to_model):
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
    
    TIMESTEPS = 25000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{sb3_algo}_c_{TIMESTEPS * iters}")

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

    episode = 10
    for episode in range(1, episode + 1):
        obs, info = env.reset()
        done = False
        truncated = False
        score = 0

        while not done and not truncated:
            action,_ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            score += reward

            env.render()
            time.sleep(1 / 240)
            

        print(f"Episode {episode}, Score: {score}")

    env.close()


if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    parser.add_argument('-c', '--cont', metavar='path_to_model')
    args = parser.parse_args()

    if args.train:
        env = gym.make(args.gymenv)
        train(env, args.sb3_algo)

    if args.test:
        if os.path.isfile(args.test):
            env = gym.make(args.gymenv)
            test(env, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')

    if args.cont:
        if os.path.isfile(args.cont):
            env = gym.make(args.gymenv)
            cont_train(env, args.sb3_algo, path_to_model=args.cont)
        else:
            print(f'{args.cont} not found.')
