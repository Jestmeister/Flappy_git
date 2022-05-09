
import unittest
import torch
from torch.autograd import Variable
import numpy as np

import agentPG



class Test(unittest.TestCase):
    def test_StartEnv(self):
        theAgent = agentPG.AgentPG()
        theAgent.Start(1, 0)

        theAgent.StartEnv()
        startScore = theAgent.env.score
        theAgent.env.score = 1000

        theAgent.StartEnv()

        self.assertEqual(theAgent.env.score, startScore)

    def test_Update(self):
        theAgent = agentPG.AgentPG()
        theAgent.Start(1, 1)
        theAgent.StartEnv()

        self.assertEqual(len(theAgent.state_pool), 0)
        self.assertEqual(len(theAgent.action_pool), 0)
        self.assertEqual(len(theAgent.reward_pool), 0)

        theAgent.Update()

        self.assertEqual(len(theAgent.state_pool), 1)
        self.assertEqual(len(theAgent.action_pool), 1)
        self.assertEqual(len(theAgent.reward_pool), 1)
    
    def test_UpdatePolicy(self):
        theAgent = agentPG.AgentPG()
        theAgent.Start(1, 4)
        theAgent.StartEnv()

        theAgent.Update()
        theAgent.Update()
        theAgent.Update()

        action1 = theAgent.action_pool[0]
        state1 = theAgent.state_pool[0]
        action2 = theAgent.action_pool[2]
        state2 = theAgent.state_pool[2]

        theAgent.reward_pool = [100000]
        theAgent.reward_pool.append(0)
        theAgent.reward_pool.append(1)

        theAgent.UpdatePolicy(1)

        self.assertAlmostEqual(action1, theAgent.policy_net(state1).item(), 2)
        self.assertNotAlmostEqual(action2, theAgent.policy_net(state2).item(), 2)


if (__name__ == '__main__'):
    unittest.main()