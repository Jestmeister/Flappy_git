
import agentPG



def main():
    # Parameters
    num_episode = 5000
    batch_size = 100
    learning_rate = 0.01
    gamma = 0.99

    theAgent = agentPG.AgentPG()

    theAgent.Start(learning_rate)

    for e in range(num_episode):
        theAgent.StartEnv()

        running = True
        while(running):
            running = theAgent.Update()

        # Update policy
        if e > 0 and e % batch_size == 0:
            theAgent.UpdatePolicy(gamma)


if __name__ == '__main__':
    main()