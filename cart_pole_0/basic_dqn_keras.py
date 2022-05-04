import gym
import torch

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2

import matplotlib.pyplot as plt


env = gym.make('CartPole-v0')
target = 500  # maximum number of iterations

input_dim = 4
n_actions = 2


class Agent:

    def __init__(self):
        self.memory = []

        self.model = Sequential()
        self.model.add(Dense(64, input_dim = input_dim , activation = 'relu'))
        self.model.add(Dense(32, activation = 'relu'))
        self.model.add(Dense(n_actions, activation = 'linear'))
        self.model.compile(optimizer=adam_v2.Adam(), loss = 'mse')

    def remember_episode(self, episode):
        self.memory.append(episode)

    def calc_rewards(self, rewards):
        if not rewards:
            return 0

        return rewards[0] + gamma * self.calc_rewards(rewards[1:])

    def learn_episode(self):
        all_states = []
        all_rewards = []

        for episode in self.memory:
            states = np.array([m['state'] for m in episode])
            actions = [m['action'] for m in episode]
            rewards = [m['reward'] for m in episode]

            estimated_reward = self.model.predict(states)
            last = len(states) - 1
            target_reward = -1 if last < target else 1

            for i in range(len(states)):
                action = actions[last - i]
                target_reward = rewards[last - i] + gamma * target_reward
                estimated_reward[last - i][action] = target_reward

            all_states.append(states)
            all_rewards.append(estimated_reward)

        all_states = np.concatenate([state for state in all_states])
        all_rewards = np.concatenate([reward for reward in all_rewards])

        self.model.fit(all_states, all_rewards, epochs=1, verbose=0)

    def get_action(self, state):
        probs = self.model.predict(state.reshape(1, 4))

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(probs)

        return action


n_episodes = 300
gamma = 0.9
epsilon = 1

agent = Agent()
max_result = 0

results_diagram = []

for n in range(n_episodes):
    state = env.reset()
    env._max_episode_steps = target

    done = False
    result = 0
    episode = []

    while not done:
        # env.render()

        action = agent.get_action(state)
        new_state, reward, done, info = env.step(action)

        episode.append({
            'state': state,
            'action': action,
            'reward': reward
        })

        # Update state
        state = new_state

        # Decrease epsilon - we will make less and less random actions
        if epsilon > 0.01:
            epsilon -= 0.001

        result += 1

    agent.remember_episode(episode)

    agent.learn_episode()

    results_diagram.append(result)

    max_result = max(max_result, result)
    print(f'Result: {result}, Max Result: {max_result}')

    if n>0 and n % 30 == 0:
        # TODO: make sure we get good reward for the last steps
        plt.plot(results_diagram)
        plt.show()
