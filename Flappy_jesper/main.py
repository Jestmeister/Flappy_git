import random  # For generating random numbers
import sys
from turtle import isvisible  # We will use sys.exit to exit the program
import pygame
from pygame.locals import *
from scipy.fftpack import diff  # Basic pygame imports
from environment import environment
from agent import DQN, DQNagent
import copy as cp
import torch

# Global Variables for the game
FPS = 32
scr_width = 289
scr_height = 511
display_screen_window = pygame.display.set_mode((scr_width, scr_height))
play_ground = scr_height * 0.8
game_image = {}
game_audio_sound = {}
player = 'images/bird.png'
bcg_image = 'images/background.png'
pipe_image = 'images/pipe.png'


def welcome_main_screen():
    """
    Shows welcome images on the screen
    """

    p_x = int(scr_width / 5)
    p_y = int((scr_height - game_image['player'].get_height()) / 2)
    msgx = int((scr_width - game_image['message'].get_width()) / 2)
    msgy = int(scr_height * 0.13)
    b_x = 0
    while True:
        for event in pygame.event.get():
            # if user clicks on cross button, close the game
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

            # If the user presses space or up key, start the game for them
            elif event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                return
            else:
                display_screen_window.blit(game_image['background'], (0, 0))
                display_screen_window.blit(game_image['player'], (p_x, p_y))
                display_screen_window.blit(game_image['message'], (msgx, msgy))
                display_screen_window.blit(game_image['base'], (b_x, play_ground))
                pygame.display.update()
                time_clock.tick(FPS)

#2 be moved???
def main_gameplay(game):
    score = 0
    p_x = int(scr_width / 5)
    p_y = int(scr_width / 2)
    b_x = 0


    n_pip1 = get_Random_Pipes()
    n_pip2 = get_Random_Pipes()
    #n_pip2 = cp.deepcopy(n_pip1)

    up_pips = [
        {'x': scr_width + 200, 'y': n_pip1[0]['y']},
        {'x': scr_width + 200 + (scr_width / 2), 'y': n_pip2[0]['y']},
    ]

    low_pips = [
        {'x': scr_width + 200, 'y': n_pip1[1]['y']},
        {'x': scr_width + 200 + (scr_width / 2), 'y': n_pip2[1]['y']},
    ]

    pip_Vx = -4

    p_vx = -9
    p_mvx = 10
    p_mvy = -8
    p_accuracy = 1

    p_flap_accuracy = -8
    p_flap = False

    while True:
        #print(up_pips)
        #print(low_pips)
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if p_y > 0:
                    p_vx = p_flap_accuracy
                    p_flap = True
                    game_audio_sound['wing'].play()

        cr_tst = is_Colliding(p_x, p_y, up_pips,
                              low_pips)
        if cr_tst:
            return


        p_middle_positions = p_x + game_image['player'].get_width() / 2
        for pipe in up_pips:
            pip_middle_positions = pipe['x'] + game_image['pipe'][0].get_width() / 2
            if pip_middle_positions <= p_middle_positions < pip_middle_positions + 4:
                score += 1
                print(f"Your score is {score}")
                game_audio_sound['point'].play()

        if p_vx < p_mvx and not p_flap:
            p_vx += p_accuracy

        if p_flap:
            p_flap = False
        p_height = game_image['player'].get_height()
        p_y = p_y + min(p_vx, play_ground - p_y - p_height)


        for pip_upper, pip_lower in zip(up_pips, low_pips):
            pip_upper['x'] += pip_Vx
            pip_lower['x'] += pip_Vx


        if 0 < up_pips[0]['x'] < 5:
            new_pip = get_Random_Pipes()
            up_pips.append(new_pip[0])
            low_pips.append(new_pip[1])


        if up_pips[0]['x'] < -game_image['pipe'][0].get_width():
            up_pips.pop(0)
            low_pips.pop(0)


        display_screen_window.blit(game_image['background'], (0, 0))
        for pip_upper, pip_lower in zip(up_pips, low_pips):
            display_screen_window.blit(game_image['pipe'][0], (pip_upper['x'], pip_upper['y']))
            display_screen_window.blit(game_image['pipe'][1], (pip_lower['x'], pip_lower['y']))

        display_screen_window.blit(game_image['base'], (b_x, play_ground))
        display_screen_window.blit(game_image['player'], (p_x, p_y))
        d = [int(x) for x in list(str(score))]
        w = 0
        for digit in d:
            w += game_image['numbers'][digit].get_width()
        Xoffset = (scr_width - w) / 2

        for digit in d:
            display_screen_window.blit(game_image['numbers'][digit], (Xoffset, scr_height * 0.12))
            Xoffset += game_image['numbers'][digit].get_width()
        pygame.display.update()
        time_clock.tick(FPS)


def is_Colliding(p_x, p_y, up_pipes, low_pipes):
    if p_y > play_ground - 25 or p_y < 0:
        game_audio_sound['hit'].play()
        return True

    for pipe in up_pipes:
        pip_h = game_image['pipe'][0].get_height()
        if (p_y < pip_h + pipe['y'] and abs(p_x - pipe['x']) < game_image['pipe'][0].get_width() - 20):
            game_audio_sound['hit'].play()
            return True

    for pipe in low_pipes:
        if (p_y + game_image['player'].get_height() > pipe['y']) and abs(p_x - pipe['x']) < \
                game_image['pipe'][0].get_width() -20:  ########## CHANGE ###############
            game_audio_sound['hit'].play()
            return True

    return False


def get_Random_Pipes():
    """
    Generate positions of two pipes(one bottom straight and one top rotated ) for blitting on the screen
    """
    pip_h = game_image['pipe'][0].get_height()
    off_s = scr_height / 3
    off_s = int(scr_height / 2.5)
    yes2 = off_s + random.randrange(0, int(scr_height - game_image['base'].get_height() - 1.2 * off_s))
    #yes2 = off_s
    pipeX = scr_width + 10
    #pipeX = 2*scr_width
    y1 = pip_h - yes2 + off_s
    pipe = [
        {'x': pipeX, 'y': -y1},  # upper Pipe
        {'x': pipeX, 'y': yes2}  # lower Pipe
    ]
    return pipe

def NN(NNInput):
    print(NNInput)

    probability = 0.1
    return random.random() < probability

def read_action(action):
        if action == 0:
            return False
        else:
            return True

if __name__ == "__main__":

    isHumanPlayer = False
    isVisual = True

    pygame.init()
    time_clock = pygame.time.Clock()
    pygame.display.set_caption('Flappy Bird Game')
    game_image['numbers'] = (
        pygame.image.load('images/0.png').convert_alpha(),
        pygame.image.load('images/1.png').convert_alpha(),
        pygame.image.load('images/2.png').convert_alpha(),
        pygame.image.load('images/3.png').convert_alpha(),
        pygame.image.load('images/4.png').convert_alpha(),
        pygame.image.load('images/5.png').convert_alpha(),
        pygame.image.load('images/6.png').convert_alpha(),
        pygame.image.load('images/7.png').convert_alpha(),
        pygame.image.load('images/8.png').convert_alpha(),
        pygame.image.load('images/9.png').convert_alpha(),
    )

    game_image['message'] = pygame.image.load('images/message.png').convert_alpha()
    game_image['base'] = pygame.image.load('images/base.png').convert_alpha()
    game_image['pipe'] = (pygame.transform.rotate(pygame.image.load(pipe_image).convert_alpha(), 180),
                          pygame.image.load(pipe_image).convert_alpha()
                          )

    # Game sounds
    game_audio_sound['die'] = pygame.mixer.Sound('sounds/die.wav')
    game_audio_sound['hit'] = pygame.mixer.Sound('sounds/hit.wav')
    game_audio_sound['point'] = pygame.mixer.Sound('sounds/point.wav')
    game_audio_sound['swoosh'] = pygame.mixer.Sound('sounds/swoosh.wav')
    game_audio_sound['wing'] = pygame.mixer.Sound('sounds/wing.wav')

    game_image['background'] = pygame.image.load(bcg_image).convert()
    game_image['player'] = pygame.image.load(player).convert_alpha()

    pipe_width = game_image['pipe'][0].get_width()
    pipe_height = game_image['pipe'][0].get_height()
    player_width = game_image['player'].get_width()
    player_height = game_image['player'].get_height()
    base_height = game_image['base'].get_height()
    #print(scr_width)
    #print(scr_height)
    #print(pipe_width)
    #print(pipe_height)
    #print(player_width)
    #print(player_height)
    #print(base_height)
    game = environment(scr_width, scr_height,pipe_width,pipe_height,player_width,player_height,base_height,difficulty=4)
    
    if isHumanPlayer:
        while True:
            welcome_main_screen()  # Shows welcome screen to the user until he presses a button
            main_gameplay(game)  # This is the main game function
    

    elif isVisual:
        old_score = 0
        #cur_state,inputs,isGameOver = game.update(False)
        agent = DQNagent(0,0)
        model = DQN(agent.n_input, agent.n_actions, agent.n_hidden)
        #model.load_state_dict(torch.load('C:/Users/jespe/Documents/GitHub/Flappy_git/Flappy_jesper/net1.pt'))
        model.load_state_dict(torch.load('C:/Users/Jesper/OneDrive/Dokument/GitHub/Flappy_git/Flappy_jesper/net1.pt'))
        model.eval()
        game.update(False)
        while True:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()

            
            


            state = torch.tensor(game.cur_state)
            if game.isGameOver:
                pass
            action  = model(state).argmax().item()
            
            if read_action(action):
                game_audio_sound['wing'].play()
            if game.score != old_score:
                old_score = game.score
                print(f"Your score is {old_score}")
                game_audio_sound['point'].play()

            game.update(read_action(action))
            
            display_screen_window.blit(game_image['background'], (0, 0))
            for pip_upper, pip_lower in zip(game.up_pips, game.low_pips):
                display_screen_window.blit(game_image['pipe'][0], (pip_upper['x'], pip_upper['y']))
                display_screen_window.blit(game_image['pipe'][1], (pip_lower['x'], pip_lower['y']))

            display_screen_window.blit(game_image['base'], (game.b_x, play_ground))
            display_screen_window.blit(game_image['player'], (game.p_x, game.p_y))
            d = [int(x) for x in list(str(game.score))]
            w = 0
            for digit in d:
                w += game_image['numbers'][digit].get_width()
            Xoffset = (scr_width - w) / 2

            for digit in d:
                display_screen_window.blit(game_image['numbers'][digit], (Xoffset, scr_height * 0.12))
                Xoffset += game_image['numbers'][digit].get_width()

            pygame.display.update()
            time_clock.tick(FPS)

    
    else:
        '''
        cur_state,inputs,isGameOver = game.update(False)
        agent = DQN(len(cur_state), 2, n_episodes=50)
        while True:
            #jump = agent.train_step(inputs, isGameOver)
            jump = NN(cur_state)
            cur_state,inputs,isGameOver = game.update(jump)
            #Make main game play with read variables
        '''
        agent = DQNagent(n_episodes=100,start_difficulty=0)
        agent.train()