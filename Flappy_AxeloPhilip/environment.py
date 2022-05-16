
import random  # For generating random numbers
import sys
from turtle import done  # We will use sys.exit to exit the program
import pygame
from pygame.locals import *  # Basic pygame imports

import torch
import numpy as np



class Game:
    def __init__(self):
        self.FPS = 32

        self.scr_width = 289
        self.scr_height = 511
        self.display_screen_window = pygame.display.set_mode((self.scr_width, self.scr_height))
        self.play_ground = self.scr_height * 0.8


        self.game_image = {}
        self.game_audio_sound = {}
        self.player = 'images/bird.png'
        self.bcg_image = 'images/background.png'
        self.pipe_image = 'images/pipe.png'
        

        pygame.init()
        self.time_clock = pygame.time.Clock()
        pygame.display.set_caption('Flappy Bird Game')
        self.game_image['numbers'] = (
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

        self.game_image['message'] = pygame.image.load('images/message.png').convert_alpha()
        self.game_image['base'] = pygame.image.load('images/base.png').convert_alpha()
        self.game_image['pipe'] = (pygame.transform.rotate(pygame.image.load(self.pipe_image).convert_alpha(), 180),
                                pygame.image.load(self.pipe_image).convert_alpha()
                                )

        # Game sounds
        self.game_audio_sound['die'] = pygame.mixer.Sound('sounds/die.wav')
        self.game_audio_sound['hit'] = pygame.mixer.Sound('sounds/hit.wav')
        self.game_audio_sound['point'] = pygame.mixer.Sound('sounds/point.wav')
        self.game_audio_sound['swoosh'] = pygame.mixer.Sound('sounds/swoosh.wav')
        self.game_audio_sound['wing'] = pygame.mixer.Sound('sounds/wing.wav')

        self.game_image['background'] = pygame.image.load(self.bcg_image).convert()
        self.game_image['player'] = pygame.image.load(self.player).convert_alpha()

    
    def Start(self, displayGameInput, limitFPSInput, start_difficulty):
        self.displayGame = displayGameInput
        self.limitFPS = limitFPSInput
        self.difficulty = start_difficulty

        self.score = 0
        self.p_x = int(self.scr_width / 5)
        self.p_y = int(self.scr_width / 2)
        self.b_x = 0


        self.n_pip1 = self.get_Random_Pipes()
        self.n_pip2 = self.get_Random_Pipes()


        self.up_pips = [
            {'x': self.scr_width + 200, 'y': self.n_pip1[0]['y']},
            {'x': self.scr_width + 200 + (self.scr_width / 2), 'y': self.n_pip2[0]['y']},
        ]

        self.low_pips = [
            {'x': self.scr_width + 200, 'y': self.n_pip1[1]['y']},
            {'x': self.scr_width + 200 + (self.scr_width / 2), 'y': self.n_pip2[1]['y']},
        ]

        self.pip_Vx = -4

        self.p_vx = -9
        self.p_mvx = 10
        self.p_mvy = -8
        self.p_accuracy = 1

        self.p_flap_accuracy = -8
        self.p_flap = False

        self.gameover = False

        # x cordinate for NN
        self.x = 0

        return self.game_State(True)


    def Update(self, jump):
        if jump == 1:
            if self.p_y > 0:
                self.p_vx = self.p_flap_accuracy
                self.p_flap = True
                self.game_audio_sound['wing'].play()

        cr_tst = self.is_Colliding(self.p_x, self.p_y, self.up_pips,
                              self.low_pips)
        if cr_tst:
            self.gameover = True


        self.p_middle_positions = self.p_x + self.game_image['player'].get_width() / 2
        for pipe in self.up_pips:
            self.pip_middle_positions = pipe['x'] + self.game_image['pipe'][0].get_width() / 2
            if self.pip_middle_positions <= self.p_middle_positions < self.pip_middle_positions + 4:
                self.score += 1
                #print(f"Your score is {self.score}")
                self.game_audio_sound['point'].play()

        if self.p_vx < self.p_mvx and not self.p_flap:
            self.p_vx += self.p_accuracy

        if self.p_flap:
            self.p_flap = False
        self.p_height = self.game_image['player'].get_height()
        self.p_y = self.p_y + min(self.p_vx, self.play_ground - self.p_y - self.p_height)


        for pip_upper, pip_lower in zip(self.up_pips, self.low_pips):
            pip_upper['x'] += self.pip_Vx
            pip_lower['x'] += self.pip_Vx
            self.x += self.pip_Vx


        if 0 < self.up_pips[0]['x'] < 5:
            new_pip = self.get_Random_Pipes()
            self.up_pips.append(new_pip[0])
            self.low_pips.append(new_pip[1])


        if self.up_pips[0]['x'] < -self.game_image['pipe'][0].get_width():
            self.up_pips.pop(0)
            self.low_pips.pop(0)


        self.display_screen_window.blit(self.game_image['background'], (0, 0))
        for pip_upper, pip_lower in zip(self.up_pips, self.low_pips):
            self.display_screen_window.blit(self.game_image['pipe'][0], (pip_upper['x'], pip_upper['y']))
            self.display_screen_window.blit(self.game_image['pipe'][1], (pip_lower['x'], pip_lower['y']))

        self.display_screen_window.blit(self.game_image['base'], (self.b_x, self.play_ground))
        self.display_screen_window.blit(self.game_image['player'], (self.p_x, self.p_y))
        d = [int(x) for x in list(str(self.score))]
        w = 0
        for digit in d:
            w += self.game_image['numbers'][digit].get_width()
        Xoffset = (self.scr_width - w) / 2

        for digit in d:
            self.display_screen_window.blit(self.game_image['numbers'][digit], (Xoffset, self.scr_height * 0.12))
            Xoffset += self.game_image['numbers'][digit].get_width()
        if self.displayGame:
            pygame.display.update()
        if self.limitFPS:
            self.time_clock.tick(self.FPS)

        #returns game state
        x_to_pipe = abs(self.p_x - self.up_pips[0]['x'])
        for pipe in self.up_pips:
            if pipe['x'] > self.p_x and abs(self.p_x - pipe['x']) < x_to_pipe:
                x_to_pipe = abs(self.p_x - pipe['x'])

        return self.game_State(False)
    
    def game_State(self, only_state):
        x_to_pipe = abs(self.p_x - self.low_pips[0]['x'])
        y_of_pipe = self.low_pips[0]['y']
        for pipe in self.low_pips:
            if pipe['x'] > self.p_x and abs(self.p_x - pipe['x']) < x_to_pipe:
                x_to_pipe = abs(self.p_x - pipe['x'])
                y_of_pipe = pipe['y']

        #return [x_to_pipe, self.p_y, y_of_pipe, self.p_vx]
        next_state = torch.tensor([[x_to_pipe, self.p_y, y_of_pipe, self.x]], dtype=torch.float32)
        if only_state:
            return next_state

        reward = torch.tensor([[self.score + 1]], dtype=torch.float32)

        done = self.gameover

        return next_state, reward, done
        
    def is_Colliding(self, p_x, p_y, up_pipes, low_pipes):
        if p_y > self.play_ground - 25 or p_y < 0:
            self.game_audio_sound['hit'].play()
            return True

        for pipe in up_pipes:
            pip_h = self.game_image['pipe'][0].get_height()
            if (p_y < pip_h + pipe['y'] and abs(p_x - pipe['x']) < self.game_image['pipe'][0].get_width()):
                self.game_audio_sound['hit'].play()
                return True

        for pipe in low_pipes:
            if (p_y + self.game_image['player'].get_height() > pipe['y']) and abs(p_x - pipe['x']) < \
                    self.game_image['pipe'][0].get_width():
                self.game_audio_sound['hit'].play()
                return True

        return False


    def get_Random_Pipes(self):
        """
        Generate positions of two pipes(one bottom straight and one top rotated ) for blitting on the screen
        """

        pip_h = self.game_image['pipe'][0].get_height()
        base_height = self.game_image['base'].get_height()
        # Constant hight on pipes
        if self.difficulty == 0:
            off_s = self.scr_height / 2
            yes2 = off_s
            pipeX = self.scr_width + 10
        # Random pipes with big gaps and far apart
        elif self.difficulty == 1:
            off_s = self.scr_height / 2
            yes2 = off_s + random.randrange(0, int(self.scr_height - base_height - 1.2 * off_s))
            pipeX = 1.5 * self.scr_width  # must be int???
        # Random pipes with big gaps
        elif self.difficulty == 2:
            off_s = self.scr_height / 2
            yes2 = off_s + random.randrange(0, int(self.scr_height - base_height - 1.2 * off_s))
            pipeX = self.scr_width + 10
        # Random pipes smaller gap
        elif self.difficulty == 3:
            off_s = np.floor(self.scr_height / 2.4)
            yes2 = off_s + random.randrange(0, int(self.scr_height - base_height - 1.2 * off_s))
            pipeX = self.scr_width + 10
        # Real deal
        elif self.difficulty == 4:
            off_s = self.scr_height / 3
            yes2 = off_s + random.randrange(0, int(self.scr_height - base_height - 1.2 * off_s))
            pipeX = self.scr_width + 10
        y1 = pip_h - yes2 + off_s
        pipe = [
            {'x': pipeX, 'y': -y1},  # upper Pipe
            {'x': pipeX, 'y': yes2}  # lower Pipe
        ]
        return pipe