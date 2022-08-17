import gym
from gym import spaces
import numpy as np

from stable_baselines3.common.env_checker import check_env

from main import MainPage


class ZeroDummyEnv(gym.Env):
    """ Custom Environment that follows gym interface. """
    metadata = {'render.modes': ['human']}
    all_actions = {}
    total_reward = 0
    _max_episode_steps = 100

    def __init__(self):
        super(ZeroDummyEnv, self).__init__()
        self.action_space = spaces.Discrete(20)
        self.observation_space = spaces.Box(low=0, high=4, shape=(20,), dtype=np.uint8)
        """
            spaces.Dict({
            'visual': spaces.Box(low=0, high=1, shape=(40, 30), dtype=np.uint8),
            'history': spaces.Box(low=0, high=22, shape=(self._max_episode_steps,), dtype=np.uint8)
        })
        """

        self.page = MainPage()
        self.all_actions = self.page.get_actions()
        self.history = []
        self.human_history = []

    def step(self, action):
        action_name = self.all_actions[action].name

        done, reward = self.all_actions[action].perform()

        self.history.append(action)
        self.human_history.append(action_name)

        state = self.get_state()
        self.all_actions = self.page.get_actions()

        # Additional reward for double clicks:
        if len(self.human_history) >= 2 and self.human_history[-2] == action_name:
            if self.human_history.count(action_name) == 2:
                reward += self.all_actions[action].double_click_reward

        # Prevent agent to stick on the same element forever:
        if self.human_history.count(action_name) > 2:
            reward = -0.001

        # Reward original actions:
        if action_name not in self.human_history[:-1]:
            if action_name != 'nothing':
                reward = reward + 0.1

        self.total_reward += reward

        if len(self.history) >= self._max_episode_steps:
            done = True

        return state, reward, done, {}

    def get_state(self):
        # 1 - we already did it before
        # 2 - we did it once on last step
        # 3 - we did it many times already
        # 4 - new action, we didn't do it yet
        # 0 - do not touch, empty element
        #

        """
        image = np.zeros((40, 30), dtype=np.uint8)

        for a in self.all_actions:
            if a.y_start:
                for y in range(a.y_start, a.y_end+1):
                    for x in range(a.x_start, a.x_end+1):
                        image[y][x] = 255
        """

        state = np.zeros((20,), dtype=np.uint8)  # 0 - do not touch, empty element

        for i, a in enumerate(self.all_actions):
            status = 0

            if a.name in self.human_history:
                if self.human_history.count(a.name) > 1:
                    status = 3  # 3 - we did it many times already
                elif self.human_history.count(a.name) == 1:
                    if len(self.human_history) > 0 and self.human_history[-1] == a.name:
                        status = 2  # 2 - we did it once on last step
                    else:
                        status = 1  # 1 - we already did it before
            else:
                if a.name != 'nothing':
                    status = 4  # 4 - new action, we didn't do it yet

            if a.name == 'nothing':
                status = 0

            state[i] = status

        return state  # .astype(np.uint8)

    def reset(self):
        self.page = MainPage()
        self.all_actions = self.page.get_actions()
        self.history = []
        self.human_history = []
        self.total_reward = 0

        state = self.get_state()

        return state

    def render(self, mode='human', close=False):
        print('Env data:')
        print(self.get_state())
        print(self.total_reward, self.human_history)

        # print(self.all_actions)

        """
        import imageio
        state = self.get_state()
        imageio.imwrite('state.png', state['visual'])

        print(state['visual'].astype(np.uint8))
        """

"""
env = ZeroDummyEnv()
env.reset()
print([a.name for a in env.all_actions])
env.step(4)
env.render()

check_env(env)
"""

env = ZeroDummyEnv()
check_env(env)


"""
env = ZeroDummyEnv()
env.reset()
env.render()
print([a.name for a in env.all_actions])
env.step(19)
env.render()
print([a.name for a in env.all_actions])
"""
