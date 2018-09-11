from random import random, randint
import numpy as np


class CatcherState:

    def __init__(self, state):
        self.ver_distance = int(state['fruit_y']) + 28
        self.hor_distance = int(state['fruit_x']) - int(state['player_x']) + 64

    def __eq__(self, other):
        return self.ver_distance == other.ver_distance and self.hor_distance == other.hor_distance


class CatcherAgent:

    def __init__(self, action_set, learning_ratio=0.01, gama=0.5, load_from_file=False, file_name=''):
        self.action_set = action_set
        self.learning_ratio = learning_ratio
        self.gama = gama
        if load_from_file:
            self.q_func = np.load(file_name + '_catcher.npy')
        else:
            self.q_func = np.zeros((3, 94, 128))

    def __choose_best_state_action(self, state):
        best_state = None
        best_action = None
        best_reward = 0
        for x in range(0, len(self.action_set)):
            reward = self.q_func[x, state.ver_distance, state.hor_distance]

            if best_state is None or reward > best_reward:
                best_state = state
                best_action = self.action_set[x]
                best_reward = reward

        return best_action

    def __get_action_index(self, action):
        for x in range(0, len(self.action_set)):
            if action == self.action_set[x]:
                return x
        return None

    def pick_action(self, state):
        catcher_state = CatcherState(state)

        best_action = self.__choose_best_state_action(catcher_state)

        if random() < self.learning_ratio:
            best_action = self.action_set[randint(0, len(self.action_set)-1)]

        return best_action

    def update_q_function(self, state, action, reward, next_state):
        state = CatcherState(state)
        next_state = CatcherState(next_state)

        action_index = self.__get_action_index(action)
        next_action_index = self.__get_action_index(self.__choose_best_state_action(next_state))

        if action_index is None or next_action_index is None:
            raise Exception('Action not included in action set')

        now_q_func = self.q_func[action_index, state.ver_distance, state.hor_distance]
        next_q_func = self.q_func[next_action_index, next_state.ver_distance, next_state.hor_distance]
        expected_reward = reward + self.gama*next_q_func
        # print(expected_reward)
        self.q_func[action_index, state.ver_distance, state.hor_distance] = \
            now_q_func + self.learning_ratio*(expected_reward - now_q_func)

    def save_q_func_on_file(self, file_name):
        np.save(file_name + '_catcher', self.q_func)
