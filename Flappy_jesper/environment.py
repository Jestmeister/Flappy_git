import random  # For generating random numbers

class environment:

    def __init__(self, scr_width, scr_height,pipe_width,pipe_height,player_width,player_height,base_height):
        self.isGameOver = False
        
        self.scr_height = scr_height
        self.scr_width = scr_width
        
        #self.open_ratio = open_ratio

        self.pipe_width = pipe_width
        self.pipe_height = pipe_height
        self.player_width = player_width
        self.player_height = player_height
        self.base_height = base_height
        
        self.scr_width = scr_width
        self.scr_height = scr_height
        self.play_ground = scr_height * 0.8

        self.score = 0
        self.p_x = int(scr_width / 5)
        self.p_y = int(scr_width / 2)
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

        self.game_State()
        self.cur_state = [self.x_to_pipe, self.p_y, self.y_of_pipe, self.p_vx]
        
    def is_Colliding(self,p_x, p_y, up_pipes, low_pipes):
        if p_y > self.play_ground - 25 or p_y < 0:
            return True

        for pipe in up_pipes:
            pip_h = self.pipe_height
            if (p_y < pip_h + pipe['y'] and abs(p_x - pipe['x']) < self.pipe_width):
                return True

        for pipe in low_pipes:
            if (p_y + self.player_height > pipe['y']) and abs(p_x - pipe['x']) < \
                    self.pipe_width:
                return True

        return False


    def get_Random_Pipes(self):
        """
        Generate positions of two pipes(one bottom straight and one top rotated ) for blitting on the screen
        """
        pip_h = self.pipe_height
        off_s = self.scr_height / 3
        yes2 = off_s + random.randrange(0, int(self.scr_height - self.base_height - 1.2 * off_s))
        pipeX = self.scr_width + 10
        y1 = pip_h - yes2 + off_s
        pipe = [
            {'x': pipeX, 'y': -y1},  # upper Pipe
            {'x': pipeX, 'y': yes2}  # lower Pipe
        ]
        return pipe

    def update(self,jump):
        
        if self.isGameOver:
            self.restart()
            self.isGameOver = False

        if jump:
            if self.p_y > 0:
                self.p_vx = self.p_flap_accuracy
                self.p_flap = True
                print('jumpin')
                #game_audio_sound['wing'].play()

        cr_tst = self.is_Colliding(self.p_x, self.p_y, self.up_pips,
                              self.low_pips)
        if cr_tst:
            #return
            print('Restarting')
            #self.restart()
            isGameOver = True
            return self.game_State(), [self.score,self.up_pips,self.low_pips,self.b_x,self.p_x,self.p_y], isGameOver


        p_middle_positions = self.p_x + self.player_width / 2
        for pipe in self.up_pips:
            pip_middle_positions = pipe['x'] + self.pipe_width / 2
            if pip_middle_positions <= p_middle_positions < pip_middle_positions + 4:
                self.score += 1
                print(f"Your score is {self.score}")
                #game_audio_sound['point'].play()

        if self.p_vx < self.p_mvx and not self.p_flap:
            self.p_vx += self.p_accuracy

        if self.p_flap:
            self.p_flap = False
        #p_height = self.player_height
        self.p_y = self.p_y + min(self.p_vx, self.play_ground - self.p_y - self.player_height)


        for pip_upper, pip_lower in zip(self.up_pips, self.low_pips):
            pip_upper['x'] += self.pip_Vx
            pip_lower['x'] += self.pip_Vx


        if 0 < self.up_pips[0]['x'] < 5:
            new_pip = self.get_Random_Pipes()
            self.up_pips.append(new_pip[0])
            self.low_pips.append(new_pip[1])


        if self.up_pips[0]['x'] < -self.pipe_width:
            self.up_pips.pop(0)
            self.low_pips.pop(0)

        #return self.game_State(), [self.score,self.up_pips,self.low_pips,self.b_x,self.p_x,self.p_y], isGameOver
        self.game_State()

    def game_State(self):
        self.x_to_pipe = abs(self.p_x - self.low_pips[0]['x'])
        self.y_of_pipe = self.low_pips[0]['y']
        for pipe in self.low_pips:
            if pipe['x'] > self.p_x and abs(self.p_x - pipe['x']) < self.x_to_pipe:
                self.x_to_pipe = abs(self.p_x - pipe['x'])
                self.y_of_pipe = pipe['y']

        #return [x_to_pipe, self.p_y, y_of_pipe, self.p_vx]
        

    def restart(self):
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

     



