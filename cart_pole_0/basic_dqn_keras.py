import gym

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2
from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt


env = gym.make('CartPole-v0')
target = 500  # maximum number of iterations

input_dim = 4   # the environment returns state as a list of 4 numbers
n_actions = 2   # the agent can perform just one of two actions - go left or right


class Agent:

    def __init__(self):
        self.memory = []

        # The number of layers and activation functions were just selected randomly
        # with experimenting with different network topologies.
        # This one works fine for me - please note - when I used less layers,
        # the eduction was not stable - after several success iteration the model made mistakes again.
        # This implementation works stable, after model learned the pattern, it follows the pattern.

        self.model = Sequential()
        self.model.add(Dense(64, input_dim=input_dim, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(n_actions, activation='linear'))
        self.model.compile(optimizer=adam_v2.Adam(), loss='mse')

    def remember_episode(self, episode):
        self.memory.append(episode)

    def learn(self):
        all_states = []
        all_rewards = []

        # Agent remembers all episodes, and we calculate rewards separately for each episode
        # TODO: we can cache calculated rewards and calculate these rewards only once
        for episode in self.memory[::-1]:
            states = np.array([m['state'] for m in episode])
            actions = [m['action'] for m in episode]
            rewards = [m['reward'] for m in episode]

            estimated_reward = self.model.predict(states)

            last = len(states) - 1
            # give extra reward for the good solutions and punish for the bad solutions
            target_reward = -3 if len(states) < target else 3

            for i in range(len(states)):
                action = actions[last - i]
                target_reward = rewards[last - i] + gamma * target_reward
                estimated_reward[last - i][action] = target_reward

            all_states.append(states)
            all_rewards.append(estimated_reward)

        # Collect data from all episodes and put them to one array to pass it to fit function
        # The Agent is learning on all episodes in remembers
        all_states = np.concatenate([state for state in all_states])
        all_rewards = np.concatenate([reward for reward in all_rewards])

        self.model.fit(all_states, all_rewards, epochs=1, verbose=0, shuffle=True)

    def get_action(self, state):
        """ Get next action for the agent. """

        probs = self.model.predict(state.reshape(1, 4))

        if np.random.random() < epsilon:
            action = env.action_space.sample()  # perform random action
        else:
            action = np.argmax(probs)     # choose the "best" action

        return action


n_episodes = 500   # total number of episodes we are going to "play"
gamma = 0.99       # discount factor for the "future rewards"
epsilon = 1        # the probability of perform random actions

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
        if result > 200:
            env.render()

        action = agent.get_action(state)
        new_state, reward, done, info = env.step(action)

        # [optional] Help the agent to learn another tactic
        # if abs(new_state[3]) < 0.02:
        #    reward = 0

        episode.append({
            'state': state,
            'action': action,
            'reward': reward
        })

        # Update state
        state = new_state
        result += 1

    # Decrease epsilon - we will make less and less random actions
    if epsilon > 0:
        epsilon -= 0.01

    agent.remember_episode(episode)

    agent.learn()

    results_diagram.append(result)

    max_result = max(max_result, result)
    print(f'#{n:4}  Result: {result:4},   Max Result: {max_result:4}')

    # Show the graph of "results stability"
    if n > 0 and n % 30 == 0:
        plt.plot(results_diagram)
        plt.ylabel('Steps')
        plt.xlabel('# of iteration')
        plt.title('Total Steps done in each iteration (500 is a limit)')
        plt.show()
