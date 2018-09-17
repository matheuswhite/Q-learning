from PyGameLearningEnvironment.ple.games import Pong


class PongState:

    dims = [48, 2, 64, 2]

    def __init__(self, raw_state):
        self.ver_distance = abs(int(raw_state['ball_y']) - int(raw_state['player_y']))
        self.ver_distance_signal = 1 if int(raw_state['ball_y']) - int(raw_state['player_y']) > 0 else 0
        self.hor_distance = int(raw_state['ball_x'])
        self.ball_vel_x = 1 if int(raw_state['ball_velocity_y']) > 0 else 0
        self.index = [self.ver_distance, self.ver_distance_signal, self.hor_distance, self.ball_vel_x]


class PongEnv:

    def __init__(self):
        self.game = Pong()
        self.actions = self.game.getActions()

    def get_current_state(self):
        pong_state = PongState(self.game.getGameState())
        return pong_state.index

    def do_action(self, ple, action_index):
        ple.act(self.actions[action_index])
