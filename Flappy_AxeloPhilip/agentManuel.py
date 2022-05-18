
import environmentManuel



class AgentManuel:
    def StartAgent(self, learning_rate, learning_rate_value, gamma, start_difficulty):
        self.env = environmentManuel.Game()

        self.start_difficulty = start_difficulty

    def StartEnv(self):
        self.currentState = self.env.Start(True, True, self.start_difficulty)

    

    #plays one frame of the game and saves the state, reward and action for UpdatePolicy()
    def Update(self):
        if self.currentState[0, 1].item() > self.currentState[0, 2].item() - 75.0:
            currentAction = True
        else:
            currentAction = False

        self.currentState, currentReward, gameover = self.env.Update(currentAction)

        return not(gameover) #running