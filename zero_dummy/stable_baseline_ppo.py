# code from https://www.youtube.com/watch?v=Mut_u40Sqz4
# python3 -m venv ./venv
# source ./venv/bin/activate
# pip3 install -U pip pyglet gym stable-baselines3[extra]
# python3 stable_baseline_ppo.py
# tensorboard --logdir=./

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from dummy_env import ZeroDummyEnv


non_vectorized_env = ZeroDummyEnv()
env = DummyVecEnv([lambda: non_vectorized_env])

model = PPO('MlpPolicy', env, verbose=True, tensorboard_log='logs')  # learning_rate=0.0001
model.learn(total_timesteps=500000)

# Note;
# 0.5M is enough to train model with simple env for 30 steps
# 2M is almost enough to train model with simple env for 100 steps
# - there were some "nothing" events in the final history

evaluate_policy(model, env, n_eval_episodes=2, render=True)
