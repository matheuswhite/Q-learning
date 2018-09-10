import os
from PyGameLearningEnvironment.ple.games import Pong
from PyGameLearningEnvironment.ple import PLE
from mtw_agent import MtwAgent

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

game = Pong()

p = PLE(game, fps=30, display_screen=False, force_fps=False)
p.init()

myAgent = MtwAgent(p.getActionSet(), learning_ratio=0.1)

nb_frames = 1000
old_reward = 0
reward = 0.0
k = 1
old_score = 0
score = 0

while True:
    print('Episode {}'.format(k))
    k += 1
    for f in range(nb_frames):
        if p.game_over(): #check if the game is over
            p.reset_game()

        # get current state
        state = game.getGameState()
        # pick action based on q_function and current state
        action = myAgent.pick_action(state)
        # apply action and get reward
        reward = p.act(action)
        score = game.getScore()
        if score != old_score or reward != old_reward:
            print('Reward {}, Score {}'.format(reward, game.getScore()))
        old_score = score
        old_reward = reward
        # get next state
        next_state = game.getGameState()
        # update q_function with state, next_state, action and reward
        myAgent.update_q_function(state, action, reward, next_state)
