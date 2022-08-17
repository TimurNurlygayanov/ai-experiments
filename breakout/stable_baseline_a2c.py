# code from https://www.youtube.com/watch?v=Mut_u40Sqz4
# python3 -m venv ./venv
# source ./venv/bin/activate
# pip3 install -r requirements.txt
# python3 stable_baseline_a2c.py
# tensorboard --logdir=./

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env


env_not_vectorized = make_atari_env('Breakout-v4', n_envs=4, seed=0)
print(env_not_vectorized.observation_space)
env = VecFrameStack(env_not_vectorized, n_stack=4)

# model = A2C('MlpPolicy', env, verbose=True, tensorboard_log='logs')  # result: ~1.45
# model = PPO('CnnPolicy', env, verbose=True, tensorboard_log='logs')  # result: ~7.3

model = A2C('CnnPolicy', env, verbose=True, tensorboard_log='logs')  # result: ~5.4 - but can go higher
model.learn(total_timesteps=300000)
print('Saving...')
model.save('my_model')

# Evaluate
res = evaluate_policy(model, env, n_eval_episodes=2, render=True)
print(res)
