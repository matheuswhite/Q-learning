from random import random, randint
from utils import Qfunction, QfunctionRow


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


class PongAgent:

    def __init__(self, action_set, learning_ratio=0.01, gama=0.5):
        self.action_set = action_set
        self.learning_ratio = learning_ratio
        self.gama = gama
        self.q_func = Qfunction()

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