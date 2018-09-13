from random import random, randint
import numpy as np

'''
self.player_y = int(state['player_y'])
self.player_vel = int(state['player_velocity'])
self.cpu_y = int(state['cpu_y'])
self.ball_x = int(state['ball_x'])
self.ball_y = int(state['ball_y'])
self.ball_vel_x = int(state['ball_velocity_x'])
self.ball_vel_y = int(state['ball_velocity_y'])
'''


class PongState:

    def __init__(self, state):
        self.ver_distance = abs(int(state['ball_y']) - int(state['player_y']))
        self.ver_distance_signal = 1 if int(state['ball_y']) - int(state['player_y']) > 0 else 0
        self.hor_distance = int(state['ball_x'])
        self.ball_vel_x = 1 if int(state['ball_velocity_y']) > 0 else 0


class PongAgent:

    def __init__(self, action_set, learning_ratio=0.01, gama=0.5, epsilon=0.01, load_from_file=False, file_name=''):
        self.action_set = action_set
        self.learning_ratio = learning_ratio
        self.gama = gama
        self.epsilon = epsilon
        if load_from_file:
            self.q_func = np.load(file_name + '_pong.npy')
        else:
            self.q_func = np.zeros((3, 2, 48, 64, 2))

    def __get_reward(self, state, action_index: int):
        return self.q_func[action_index, state.ball_vel_x, state.ver_distance, state.hor_distance, state.ver_distance_signal]

    def __set_reward(self, state, action_index: int, reward):
        self.q_func[action_index, state.ball_vel_x, state.ver_distance, state.hor_distance, state.ver_distance_signal] = reward

    def __choose_best_state_action(self, state):
        best_state = None
        best_action = None
        best_reward = 0
        for x in range(0, len(self.action_set)):
            # print(state.ver_distance)
            reward = self.__get_reward(state, x)

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
        pong_state = PongState(state)

        best_action = self.__choose_best_state_action(pong_state)

        if random() < self.epsilon:
            best_action = self.action_set[randint(0, len(self.action_set)-1)]

        return best_action

    def update_q_function(self, state, action, reward, next_state):
        state = PongState(state)
        next_state = PongState(next_state)

        action_index = self.__get_action_index(action)
        next_action_index = self.__get_action_index(self.__choose_best_state_action(next_state))

        if action_index is None or next_action_index is None:
            raise Exception('Action not included in action set')

        now_q_func = self.__get_reward(state, action_index)
        next_q_func = self.__get_reward(next_state, next_action_index)
        expected_reward = reward + self.gama*next_q_func
        # print(expected_reward)
        self.__set_reward(state, action_index, now_q_func + self.learning_ratio*(expected_reward - now_q_func))

    def save_q_func_on_file(self, file_name):
        np.save(file_name + '_pong', self.q_func)
