
import agentPG
import matplotlib.pyplot as plt
import agentManuel

import agentPGCNN



def main():
    # Parameters
    num_episode = 16000
    batch_size = 1000

    learning_rate = 0.01
    learning_rate_value = 0.005
    gamma = 0.9

    start_difficulty = 0

    num_pre_train = 4000
    batch_size_after_pre_train = 500
    learning_rate_value_after_pre_train = 0.01 * learning_rate_value

    theAgent = agentPG.AgentPG()

    theAgent.StartAgent(learning_rate, learning_rate_value, gamma, start_difficulty)
    theAgent.batch_size = batch_size

    if num_pre_train > 0:
        theAgent.preTrainValueNet = True
    else:
        theAgent.preTrainValueNet = False
    plot_reward_array = []
    plot_DiscReward_array = []
    for e in range(num_episode):
        theAgent.StartEnv()

        running = True
        while(running):
            running = theAgent.Update()
        plot_reward_array.append(len(theAgent.reward))
        plot_DiscReward_array.append(len(theAgent.discountedReward))
        print(theAgent.reward)
        # print(len(theAgent.reward))
        # Update policy
        if e > 0 and e % batch_size == 0:
            print('Batch: {}'.format(e))
            theAgent.UpdatePolicy()

            if num_pre_train > e:
                theAgent.preTrainValueNet = True
            else:
                theAgent.preTrainValueNet = False
                batch_size = batch_size_after_pre_train
                theAgent.batch_size = batch_size
                learning_rate_value = learning_rate_value_after_pre_train
    print(plot_reward_array)
    plt.plot(plot_reward_array)
    plt.show()

if __name__ == '__main__':
    main()
