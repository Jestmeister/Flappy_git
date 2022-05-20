import random  # For generating random numbers
import sys
from turtle import isvisible  # We will use sys.exit to exit the program
import pygame
from pygame.locals import *
from scipy.fftpack import diff  # Basic pygame imports
from environment import environment
from agent import DQN, DQNagent
from agent_cn import DQN_cn, DQNagent_cn
import copy as cp
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import collections
#CN trained bird worked good on manual input test environment!?!?!?
from agentPG_cp import PolicyNet

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
resize = T.Compose([T.ToPILImage(),
            T.Resize(40, interpolation=Image.CUBIC),
            T.ToTensor()])

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

def main_gameplay(game):
    while True:
        flapper = False
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if game.p_y > 0:
                    flapper = True
                    game_audio_sound['wing'].play()

        p_middle_positions = game.p_x + game_image['player'].get_width() / 2
        for pipe in game.up_pips:
            pip_middle_positions = pipe['x'] + game_image['pipe'][0].get_width() / 2
            if pip_middle_positions <= p_middle_positions < pip_middle_positions + 4:
                game_audio_sound['point'].play()
        
        if game.isGameOver:
            game_audio_sound['hit'].play()
            game.restart()
            return 

        if flapper:
            flapper = False
            game.update(True)
        else:
            game.update(False)

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
        #imgdata = pygame.surfarray.array3d(display_screen_window) ######
        pygame.display.update()
        time_clock.tick(FPS)



def NN(NNInput):
    print(NNInput)

    probability = 0.1
    return random.random() < probability

def get_screen(imdata):
        screen = imdata.transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        return resize(screen).unsqueeze(0)

def read_action(action):
        if action == 0:
            return False
        else:
            return True

def SelectAction(policyNet, currentState):
    actionProbabilityDis = policyNet(currentState) #takes a state and returns policy dis

    #selects random action based on policy
    selection = random.random()
    if selection < actionProbabilityDis.item():
        return True
    else:
        return False

if __name__ == "__main__":

    isHumanPlayer = False
    testAgent = False
    testAgent_PG = True
    test_cn_agent = False

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
    
    n_test = 100
    score_ls = np.zeros(n_test)

    if isHumanPlayer:
        while True:
            welcome_main_screen()  # Shows welcome screen to the user until he presses a button
            main_gameplay(game)  # This is the main game function

    ########  DENSE #########

    elif testAgent:
        FPS = 256
        for ep in range(n_test):
            old_score = 0
            agent = DQNagent(0,0)
            model = DQN(agent.n_input, agent.n_actions, agent.n_hidden)
            #model.load_state_dict(torch.load('C:/Users/jespe/Documents/GitHub/Flappy_git/Flappy_jesper/net1.pt'))
            model.load_state_dict(torch.load('C:/Users/Jesper/OneDrive/Dokument/GitHub/Flappy_git/Flappy_jesper/net_dense.pt'))
            model.eval()
            game.update(False)
            while True:
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        pygame.quit()
                        sys.exit()
                state = torch.tensor(game.cur_state)
                if game.isGameOver:
                    score_ls[ep] = old_score
                    break
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
    




    ########## POLICY ##############

    elif testAgent_PG:
        FPS = 256
        for ep in range(n_test):
            old_score = 0
            model = PolicyNet()
            #model.load_state_dict(torch.load('C:/Users/jespe/Documents/GitHub/Flappy_git/Flappy_jesper/net1.pt'))
            model.load_state_dict(torch.load('C:/Users/Jesper/OneDrive/Dokument/GitHub/Flappy_git/Flappy_jesper/savednetn1.pt'))
            model.eval()
            game.update(False)
            while True:
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        pygame.quit()
                        sys.exit()
                state = torch.tensor(game.cur_state)
                if game.isGameOver:
                    score_ls[ep] = old_score
                    break
                #action  = model(state).argmax().item()
                
                if SelectAction(model, state):
                    game_audio_sound['wing'].play()
                if game.score != old_score:
                    old_score = game.score
                    print(f"Your score is {old_score}")
                    game_audio_sound['point'].play()

                game.update(SelectAction(model, state))
                
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



    ############ CONVOLUTION #############

    elif test_cn_agent:
        FPS = 128
        for ep in range(n_test):
            old_score = 0
            agent = DQNagent_cn(0,0)
            model = DQN_cn(agent.scr_height, agent.scr_width, agent.n_actions)
            #model.load_state_dict(torch.load('C:/Users/jespe/Documents/GitHub/Flappy_git/Flappy_jesper/net1.pt'))
            model.load_state_dict(torch.load('C:/Users/Jesper/OneDrive/Dokument/GitHub/Flappy_git/Flappy_jesper/net_cnn10.pt'))
            model.eval()
            game.update(False)

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
            imgdata = pygame.surfarray.array3d(display_screen_window) 

            last_screen = get_screen(imgdata)
            current_screen = get_screen(imgdata)

            state = current_screen - last_screen
            while True:
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        pygame.quit()
                        sys.exit()
                #state = torch.tensor(game.cur_state)
                if game.isGameOver:
                    score_ls[ep] = old_score
                    game_audio_sound['hit'].play()
                    break
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
                imgdata = pygame.surfarray.array3d(display_screen_window)
                last_screen = cp.deepcopy(current_screen)
                current_screen = get_screen(imgdata) 
                pygame.display.update()
                time_clock.tick(FPS)



    score_ls = score_ls.astype(int)
    print(f'Best score obtained durring testing: {max(score_ls)}')
    plt.figure()
    plt.plot(score_ls)
    plt.xlabel('Episode Number')
    plt.ylabel('Score')
    
    plt.figure()
    #plt.hist(score_ls, bins=max(score_ls))
    d = collections.Counter(score_ls)
    plt.bar(d.keys(), d.values())
    
    plt.xlabel('Score')
    plt.ylabel('Frequency')

    plt.show()