
import agentPG

def main():
    # Parameters
    num_episode = 100000
    batch_size = 1000
    learning_rate = 0.01
    learning_rate_value = 0.0002
    gamma = 0.9
    start_difficulty = 0
    num_pre_train = 20000

    theAgent = agentPG.AgentPG()

    theAgent.StartAgent(learning_rate, learning_rate_value, gamma, start_difficulty)

    if num_pre_train > 0:
        theAgent.preTrainValueNet = True
    else:
        theAgent.preTrainValueNet = False

    for e in range(num_episode):
        theAgent.StartEnv()

        running = True
        while(running):
            running = theAgent.Update()

        # Update policy
        if e > 0 and e % batch_size == 0:
            print("")
            print('Batch: {}'.format(e))
            theAgent.UpdatePolicy()

            if num_pre_train > e:
                theAgent.preTrainValueNet = True
            else:
                theAgent.preTrainValueNet = False



if __name__ == '__main__':
    main()