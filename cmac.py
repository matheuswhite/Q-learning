import numpy as np
from random import randint, random


class CMACFrame:

    def __init__(self, dims: list, offsets: list):
        self.frame = np.zeros(dims)
        self.dims = dims
        self.offsets = offsets

    def __get_index(self, tile: list):
        index = []
        for x in range(len(self.offsets)):
            i = tile[x] - self.offsets[x]
            if i < 0 or i >= self.dims[x]:
                return None
            index.append(i)
        return index

    def update_tile(self, tile: list, new_value: float):
        index = self.__get_index(tile)
        if index:
            self.frame[index] = new_value

    def get_tile(self, tile: list):
        index = self.__get_index(tile)
        if index:
            return self.frame[index]
        else:
            return 0


class CMAC:

    def __init__(self, game_env, states_dims: list, frames: int, actions: int, epsilon=0.01, learning_rate=0.1 ,gama=0.9):
        self.game_env = game_env
        self.epsilon = epsilon
        self.gama = gama
        self.learning_rate = learning_rate
        self.states_dims = states_dims
        self.actions = list(range(actions))
        self.frames = []

        self.dims = self.states_dims + [len(self.actions)]

        for x in range(frames):
            offsets = self.__gen_rand_offsets()
            self.frames.append(CMACFrame(self.dims, offsets))

    def __gen_rand_offsets(self):
        offsets = []
        for d in self.states_dims:
            offsets.append(randint(0, d-1))
        offsets.append(0)
        return offsets

    def __choose_best_state_action(self, state: list):
        best_action = 0
        for x in range(1, len(self.actions)):
            best_index = state + [best_action]
            sum_best = sum(frame.get_tile(best_index) for frame in self.frames)
            index = state + [x]
            summ = sum(frame.get_tile(index) for frame in self.frames)
            if summ > sum_best:
                best_action = x
        return best_action

    def pick_action(self, state: list):
        best_action = self.__choose_best_state_action(state)

        if random() < self.epsilon:
            best_action = self.actions[randint(0, len(self.actions) - 1)]

        return best_action

    def update_q_function(self, state: list, action: int, reward: float, next_state: list):
        next_action = self.__choose_best_state_action(next_state)

        now_index = state + [action]
        now_q_func = sum(frame.get_tile(now_index) for frame in self.frames)
        next_index = next_state + [next_action]
        next_q_func = sum(frame.get_tile(next_index) for frame in self.frames)

        expected_reward = reward + self.gama * next_q_func
        for frame in self.frames:
            frame.update_tile(now_index, now_q_func + self.learning_rate * (expected_reward - now_q_func))
