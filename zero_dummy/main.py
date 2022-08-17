import random
import uuid

avatar_history = []
current_history_position = 0


class Page:

    actions = {}

    def get_actions(self):
        return self.actions

    def step(self, action):
        pass


def click_random():
    # reward for click on random button
    return False, 0.001 * random.randint(50, 800)


def click_accessory():
    global avatar_history
    global current_history_position

    avatar_history.append(str(uuid.uuid4()))
    current_history_position = len(avatar_history) - 1

    done, reward = click_random()
    return done, 0.1 * reward


def click_close():
    return True, 0.001


def click_redo():
    global current_history_position
    global avatar_history

    if current_history_position < len(avatar_history) - 1:
        current_history_position += 1
        return False, 0.001

    return False, 0


def click_undo():
    global current_history_position
    global avatar_history

    if len(avatar_history) > 0 and current_history_position > 0:
        current_history_position -= 1
        return False, 0.001

    return False, 0


def click_save():
    if current_history_position > 0 and current_history_position < len(avatar_history):
        return False, 0.002
    if len(avatar_history) > 0:
        return False, 0.002

    return False, 0.001


def empty_function():
    return False, 0


class Action:
    action_function = lambda x: x
    name = ''
    clicks_count = 0
    double_click_reward = 0.1

    # area where the object is placed
    y_start = 0
    y_end = 0
    x_start = 0
    x_end = 0

    def __init__(self, name, action_function, box=None, double_click_reward=0.1):
        self.name = name
        self.action_function = action_function
        self.double_click_reward = double_click_reward

        if box:
            self.x_start, self.y_start, self.x_end, self.y_end = box

    def perform(self):
        self.clicks_count += 1

        res = self.action_function()
        return res


class ListsPage(Page):
    current_page = 0
    page_index = 100
    max_pages = 5

    basic_actions = []

    def __init__(self):
        self.actions = [
            Action(
                f'avatar_accessory{self.page_index + i}',
                click_accessory,
                box=[
                    3 + 8 * (i % 3),
                    22 + 9 * (i % 2),
                    3 + 8 * (i % 3) + 6,
                    22 + 9 * (i % 2) + 6,
                ],
                double_click_reward=0.001
            ) for i in range(self.max_pages * 6 - 2)
        ]

    def scroll_up(self):
        if self.current_page > 0:
            self.current_page -= 1

            return click_random()

        return False, 0

    def scroll_down(self):
        if self.current_page < self.max_pages:
            self.current_page += 1

            return click_random()

        return False, 0

    def get_actions(self):
        start = self.current_page * 6
        end = self.current_page * 6 + 7
        return self.basic_actions + self.actions[start:end]


class ShopPage(ListsPage):
    page_index = 200


class ExplorePage(ListsPage):
    page_index = 300

    basic_actions = [
        Action('random', click_random, box=[26, 16, 28, 17], double_click_reward=2),
        Action('save', click_save, box=[26, 1, 28, 2]),
        Action('redo', click_redo, box=[1, 16, 2, 17]),
        Action('undo', click_undo, box=[1, 19, 2, 20]),
    ]


class MainPage(Page):

    subpage = None
    actions = {}

    def __init__(self):
        self.subpage = ShopPage()

        self.actions = [
            Action('scroll_up', self.scroll_up),
            Action('scroll_down', self.scroll_down),
            Action('close', click_close, box=[1, 1, 2, 2]),
            Action('shop_tab', self.click_shop_tab, box=[3, 18, 5, 19], double_click_reward=0),
            Action('explore_tab', self.click_explore_tab, box=[7, 18, 9, 19], double_click_reward=0)
        ]

    def click_explore_tab(self):
        if self.subpage.__class__.__name__ == 'ExplorePage':
            return False, 0

        self.subpage = ExplorePage()
        return False, 0.1

    def click_shop_tab(self):
        if self.subpage.__class__.__name__ == 'ShopPage':
            return False, 0

        self.subpage = ShopPage()
        return False, 0.1

    def scroll_up(self):
        return self.subpage.scroll_up()

    def scroll_down(self):
        return self.subpage.scroll_down()

    def get_actions(self):
        result = self.actions + self.subpage.get_actions()
        result += [Action('nothing', lambda: (False, 0))] * (20-len(result))

        return result
