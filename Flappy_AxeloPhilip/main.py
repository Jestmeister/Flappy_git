
import agentPG

def main():
    # Parameters
    num_episode = 5000
    batch_size = 50
    learning_rate = 0.01
    gamma = 0.99
    start_difficulty = 0

    theAgent = agentPG.AgentPG()

    theAgent.StartAgent(learning_rate, gamma, start_difficulty)

    for e in range(num_episode):
        theAgent.StartEnv()

        running = True
        while(running):
            running = theAgent.Update()

        # Update policy
        if e > 0 and e % batch_size == 0:
            print(e)
            theAgent.UpdatePolicy()



if __name__ == '__main__':
    main()