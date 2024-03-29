
import agentPPO



def main():
    # Parameters
    num_episode = 30000
    batch_size = 1000
    batch_runs = 2

    learning_rate = 0.01
    learning_rate_value = 0.00005
    gamma = 0.8

    start_difficulty = 4

    num_pre_train = 0
    batch_size_after_pre_train = 200
    learning_rate_value_after_pre_train = 0.01 * learning_rate_value



    theAgent = agentPPO.AgentPPO()
    theAgent.StartAgent(learning_rate, learning_rate_value, gamma, start_difficulty)
    theAgent.batch_size = batch_size
    theAgent.batch_runs = batch_runs

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
            print('Batch: {}'.format(e))
            theAgent.UpdatePolicy()
            theAgent.UpdateOld()

        if num_pre_train > e:
            theAgent.preTrainValueNet = True
        else:
            theAgent.preTrainValueNet = False
            batch_size = batch_size_after_pre_train
            theAgent.batch_size = batch_size
            learning_rate_value = learning_rate_value_after_pre_train



if __name__ == '__main__':
    main()