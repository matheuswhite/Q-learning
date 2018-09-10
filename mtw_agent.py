from random import random, randint


class PongState:

    def __init__(self, state):
        self.player_y = int(state['player_y'])
        self.player_vel = int(state['player_velocity'])
        self.cpu_y = int(state['cpu_y'])
        self.ball_x = int(state['ball_x'])
        self.ball_y = int(state['ball_y'])
        self.ball_vel_x = int(state['ball_velocity_x'])
        self.ball_vel_y = int(state['ball_velocity_y'])

    def __eq__(self, other):
        return self.player_y == other.player_y and self.player_vel == other.player_vel and self.cpu_y == other.cpu_y \
               and self.ball_x == other.ball_x and self.ball_y == other.ball_y and self.ball_vel_x == other.ball_vel_x \
               and self.ball_vel_y == other.ball_vel_y


class QfunctionRow:

    def __init__(self, state, action, reward):
        self.state = state
        self.action = action
        self.reward = reward

    def __eq__(self, other):
        return self.state == other.state and self.action == other.action

    def __gt__(self, other):
        return self.reward > other.reward

    def __lt__(self, other):
        return self.reward < other.reward


class Qfunction:

    def __init__(self, action_set_len, state_set_len):
        self.table = []
        self.action_set_len = action_set_len
        self.state_set_len = state_set_len

    def find_state_action(self, value):
        for x in range(0, len(self.table)):
            if value == self.table[x]:
                return x
        return None

    def update_expected_return(self, state, action, reward):
        row = QfunctionRow(state, action, reward)
        index = self.find_state_action(row)
        if index is not None:
            self.table[index].reward = reward
        else:
            self.table.append(row)

    def get_expected_return(self, state, action):
        row = QfunctionRow(state, action, 0)
        index = self.find_state_action(row)
        if index is not None:
            return self.table[index].reward
        else:
            return 0


class MtwAgent:

    def __init__(self, action_set, learning_ratio=0.01, gama=0.5):
        self.action_set = action_set
        self.learning_ratio = learning_ratio
        self.gama = gama
        self.q_func = Qfunction(len(action_set), 7)

    def __choose_best_row(self, state):
        best_row = None
        for action in self.action_set:
            row = QfunctionRow(state, action, 0)
            index = self.q_func.find_state_action(row)

            if best_row is None:
                best_row = row
                continue

            if index is not None and row > best_row:
                best_row = row

        return best_row

    def pick_action(self, state):
        pong_state = PongState(state)

        best_row = self.__choose_best_row(pong_state)

        if random() < self.learning_ratio:
            best_row.action = self.action_set[randint(0, 2)]

        return best_row.action

    def update_q_function(self, state, action, reward, next_state):
        state = PongState(state)
        next_state = PongState(next_state)

        current_q_func = self.q_func.get_expected_return(state, action)
        best_row = self.__choose_best_row(next_state)
        next_q_func = self.q_func.get_expected_return(best_row.state, best_row.action)
        equation = current_q_func + self.learning_ratio*(reward + self.gama*next_q_func - current_q_func)
        # print(equation)
        self.q_func.update_expected_return(state, action, equation)

'''
player_y     48    0
player_vel  3.6 -3.6
cpu_y        48    0
ball_x       64    0
ball_y       48    0
ball_vel x   36  -38
ball_vel_y   36  -36
'''