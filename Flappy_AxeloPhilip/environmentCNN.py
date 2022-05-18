
import torch
import numpy as np
import copy as cp

import environment



class GameCNN(environment.Game):
    def game_State(self):
        if self.start:
            self.current_screen = self.get_screen()

        last_screen = cp.deepcopy(self.current_screen)
        self.current_screen = self.get_screen()
        state = self.current_screen - last_screen
        
        if self.new_score:
            reward = torch.tensor([[10]], dtype=torch.float32)
        else:
            reward = torch.tensor([[1]], dtype=torch.float32)

        done = self.gameover
        
        return state, reward, done
    
    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.imgdata.transpose((2, 0, 1))
        
        
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0)