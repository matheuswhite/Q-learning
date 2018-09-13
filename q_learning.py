import os
from PyGameLearningEnvironment.ple.games import Catcher, Pong
from PyGameLearningEnvironment.ple import PLE
from catcher_agent import CatcherAgent
from pong_agent import PongAgent

show_game = True

if not show_game:
    os.putenv('SDL_VIDEODRIVER', 'fbcon')
    os.environ["SDL_VIDEODRIVER"] = "dummy"

game = Pong()
# game = Catcher()

p = PLE(game, fps=30, display_screen=show_game, force_fps=not show_game)
p.init()

myAgent = PongAgent(p.getActionSet(), load_from_file=True, file_name='q_func', gama=0.9, learning_ratio=0.1, epsilon=0.001)
# myAgent = CatcherAgent(p.getActionSet(), load_from_file=True, file_name='q_func')

nb_frames = 10000
old_reward = 0
reward = 0.0
k = 1
old_score = 0
score = 0
total_reward = 0

try:
    for f in range(50000):
        print('Episode {}'.format(k))
        k += 1
        total_reward = 0
        for f in range(nb_frames):
            if p.game_over(): #check if the game is over
                p.reset_game()

            # get current state
            state = game.getGameState()
            # pick action based on q_function and current state
            action = myAgent.pick_action(state)
            # apply action and get reward
            reward = p.act(action)
            # total_reward += reward
            score = game.getScore()
            if score != old_score and score > 0:
                print('Score {}'.format(game.getScore()))
                pass
            old_score = score
            old_reward = reward
            # get next state
            next_state = game.getGameState()
            # update q_function with state, next_state, action and reward
            myAgent.update_q_function(state, action, reward, next_state)
        # print('Total Reward: {}'.format(total_reward))
    myAgent.save_q_func_on_file('q_func')

    myAgent = PongAgent(p.getActionSet(), load_from_file=True, file_name='q_func', gama=0.9, learning_ratio=0)
    total_reward = 0
    for f in range(nb_frames):
        if p.game_over():  # check if the game is over
            p.reset_game()

        # get current state
        state = game.getGameState()
        # pick action based on q_function and current state
        action = myAgent.pick_action(state)
        # apply action and get reward
        reward = p.act(action)
        total_reward += reward
        score = game.getScore()
        if score != old_score and score > 0:
            print('Score {}'.format(game.getScore()))
            pass
        old_score = score
        old_reward = reward
        # get next state
        next_state = game.getGameState()
        # update q_function with state, next_state, action and reward
        myAgent.update_q_function(state, action, reward, next_state)
    print('Total Reward: {}'.format(total_reward))
finally:
    myAgent.save_q_func_on_file('q_func')
