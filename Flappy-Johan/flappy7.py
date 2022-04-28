import random  # For generating random numbers
import sys  # We will use sys.exit to exit the program
import pygame
from pygame.locals import *  # Basic pygame imports
import time

import numpy as np
import math
import torch
from torch import nn


torch.autograd.set_detect_anomaly(True)

#Create the network
class DeepQNN(nn.Module):
    def __init__(self):
        super(DeepQNN, self).__init__()
        self.myNetwork = nn.Sequential(
            nn.Linear(6, 6),
            nn.Linear(6, 6),
            nn.Linear(6, 2),
        )

    def forward(self, x):
        qValues = self.myNetwork(x)
        return torch.sigmoid(qValues)
 
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DeepQNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.001)
print(model)

X = torch.rand(6, device=device)
print(X)



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
print("aaa")


def welcome_main_screen():
    """
    Shows welcome images on the screen
    """
    print("bbb")
    p_x = int(scr_width / 5)
    p_y = int((scr_height - game_image['player'].get_height()) / 2)
    msgx = int((scr_width - game_image['message'].get_width()) / 2)
    msgy = int(scr_height * 0.13)
    b_x = 0
    while True:
        #print("ccc")
        for event in pygame.event.get():

            # if user clicks on cross button, close the game
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

            # If the user presses space or up key, start the game for them
            elif event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                print("ddd")
                return
            else:
                display_screen_window.blit(game_image['background'], (0, 0))
                display_screen_window.blit(game_image['player'], (p_x, p_y))
                display_screen_window.blit(game_image['message'], (msgx, msgy))
                display_screen_window.blit(game_image['base'], (b_x, play_ground))
                pygame.display.update()
                time_clock.tick(FPS)


def main_gameplay():
    score = 0
    p_x = int(scr_width / 5)
    p_y = int(scr_width / 2)
    b_x = 0


    n_pip1 = get_Random_Pipes()
    n_pip2 = get_Random_Pipes()


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

    step=0
    inputHistory = np.empty((6, 0), np.double)
    outputHistory = []
    outputHistory2 = []


    # empty_array = np.append(empty_array, np.array([column_list_1]).transpose(), axis=1)
    # print(empty_array)
    # empty_array[:, 1]

    while True:
        
        step+=1
        #time.sleep(0.1)
        FBoutput=[0,0,500,0,500,0]#bird pos,bird vel, pipe1 x, pipe1 y, pipe 2 x, pipe 2 y
        #print("\n")
       # print("Step:"+str(step))
        #print("Bird velocity:"+str(p_vx))
        FBoutput[1]=p_vx
        #print("Bird position:"+str(p_y))
        FBoutput[0]=p_y
        pipeCounter=0
        pipeOutputCounter=0
        for pipe in up_pips:
            if(pipeCounter<2):
                #print(str(up_pips[pipeCounter]['x']-p_x+game_image['pipe'][0].get_width()/2 ))
                if(up_pips[pipeCounter]['x']-p_x  >= -game_image['pipe'][0].get_width()/2):
                    #print("Pipe "+str(pipeCounter+1)+" x pos"+str(up_pips[pipeCounter]['x']))
                    FBoutput[2+2*pipeOutputCounter]=up_pips[pipeCounter]['x']
                    #print("Pipe "+str(pipeCounter+1)+" y pos"+str((up_pips[pipeCounter]['y']+low_pips[pipeCounter]['y'])/2+game_image['pipe'][0].get_height()/2))
                    FBoutput[3+2*pipeOutputCounter]=(up_pips[pipeCounter]['y']+low_pips[pipeCounter]['y'])/2+game_image['pipe'][0].get_height()/2
                    pipeOutputCounter+=1
                pipeCounter+=1
      
        #print(FBoutput)
        inputHistory = np.append(inputHistory, np.array([FBoutput]).transpose(), axis=1)
        #print(inputHistory[:, 0])
        NN=True
        if(NN==False):        
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                    if p_y > 0:
                        p_vx = p_flap_accuracy
                        p_flap = True
                        game_audio_sound['wing'].play()
        else:
            #ADD NN
            #print("NN run")
            Qvalues = model(torch.FloatTensor(FBoutput))
            
            if Qvalues[0]>Qvalues[1]:
                p_flap =True
                
            #print(p_flap)
           # print(Qvalues)   
            outputHistory.append(Qvalues)


            #p_flap=NN(FBoutput)
            
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()


        #Learning
    
        if(step>400):
            for i in range(step-41,step-1):
                print(inputHistory[:, i])
                UR=0.1
                for i in range(max(step-21,0),step-1):
                    UR=UR*1.3
                    Qvalues = model(torch.FloatTensor(inputHistory[:, i]))
                    Qtarget = model(torch.FloatTensor(inputHistory[:, i]))
    
                    print(Qvalues)

                    if(Qvalues[0]<0):
                        Qtarget[0]=0.0001
                    if(Qvalues[1]<0):
                        Qtarget[1]=0.0001
                    if(Qvalues[0]>Qvalues[1]):
                        print('aaaaaaaa')
                        Qtarget[0]=0.99
                    else:
                        print('bbbbbbb')
                        Qtarget[1]=0.99
                    print(Qtarget)
                    # Zero your gradients for every batch!
                    optimizer.zero_grad()
                    
                    # Compute the loss and its gradients
                    loss_fn = torch.nn.L1Loss()
                    loss = loss_fn(Qvalues, [0,1])
                    loss.backward()
                    optimizer.step()
                #print(outputHistory[i])
        #If score reward past behavior(latest 50  after first score)
        
        #Large punishment of last 50 if fail








        cr_tst = is_Colliding(p_x, p_y, up_pips, low_pips)
        if cr_tst:
            print('asdad')
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.001)                        
            X = torch.rand(6, device=device)
            X
            X[0].item()
            a=[X[0].item(),X[1].item()]
            a=[1,2,3,4,5,6]
            b=torch.FloatTensor(a)
            Qvalues = model(b)
            Qtarget=[Qvalues[0].item(),Qvalues[1].item()]
            
            #Qvalues = model(torch.FloatTensor(FBoutput))
            print(X)
            
            for d in range(0,100):
            
                Qvalues = model(X)
                print(Qvalues)
                Qtarget=torch.FloatTensor([0.9,0.1])
                print(Qtarget)
                #Qtarget=torch.tensor([1.0,0.0])
                
                # Zero your gradients for every batch!
                optimizer.zero_grad()
                
                
                # Compute the loss and its gradients
                loss_fn = torch.nn.L1Loss()
                loss = loss_fn(Qvalues, Qtarget)
                loss.backward()
                optimizer.step()
    
            #time.sleep(5)
            print(step)
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
        #game_audio_sound['hit'].play()
        return True

    for pipe in up_pipes:
        pip_h = game_image['pipe'][0].get_height()
        if (p_y < pip_h + pipe['y'] and abs(p_x - pipe['x']) < game_image['pipe'][0].get_width()):
            #game_audio_sound['hit'].play()
            return True

    for pipe in low_pipes:
        if (p_y + game_image['player'].get_height() > pipe['y']) and abs(p_x - pipe['x']) < \
                game_image['pipe'][0].get_width():
            #game_audio_sound['hit'].play()
            return True

    return False


def get_Random_Pipes():
    """
    Generate positions of two pipes(one bottom straight and one top rotated ) for blitting on the screen
    """
    pip_h = game_image['pipe'][0].get_height()
    off_s = scr_height / 3
    yes2 = off_s + random.randrange(0, int(scr_height - game_image['base'].get_height() - 1.2 * off_s))
    pipeX = scr_width + 10
    y1 = pip_h - yes2 + off_s
    pipe = [
        {'x': pipeX, 'y': -y1},  # upper Pipe
        {'x': pipeX, 'y': yes2}  # lower Pipe
    ]
    return pipe


if __name__ == "__main__":

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

    while True:
        #welcome_main_screen()  # Shows welcome screen to the user until he presses a button
        print("eee")
        main_gameplay()  # This is the main game function