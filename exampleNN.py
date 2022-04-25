
import random

import flappyNN



def NN(NNInput):
    print(NNInput)

    probability = 0.1
    return random.random() < probability



theGame = flappyNN.Game()

# theGame.Start(displayGameInput, limitFPSInput)
gameOutput = theGame.Start(False, False)

while(True):
    #theGame.Update(Jump = input aka network output should be True or False) -> Game state
    gameOutput = theGame.Update(NN(gameOutput))

    if gameOutput == None:
        break