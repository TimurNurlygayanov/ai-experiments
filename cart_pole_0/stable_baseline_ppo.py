# code from https://www.youtube.com/watch?v=Mut_u40Sqz4
# python3 -m venv ./venv
# source ./venv/bin/activate
# pip3 install -U pip pyglet gym stable-baselines3[extra]
# python3 stable_baseline_ppo.py
# tensorboard --logdir=./

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


env = gym.make('CartPole-v0')
target = 500
env._max_episode_steps = target
env = DummyVecEnv([lambda: env])

# Callbacks: 1:30 on the video
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=target, verbose=True)
eval_callback = EvalCallback(
    env,
    callback_on_new_best=stop_callback,
    eval_freq=2000,
    best_model_save_path='my_model',
    verbose=True)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='logs')
model.learn(total_timesteps=20000, callback=eval_callback)

# model.save('my_model')
# model = PPO.load('my_model', env=env)

evaluate_policy(model, env, n_eval_episodes=10, render=True)
