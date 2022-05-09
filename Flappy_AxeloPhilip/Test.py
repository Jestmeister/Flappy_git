
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

        input = np.array([1., -1., 0.5, -1.5])
        input = torch.from_numpy(input).float()
        input = Variable(input)

        action1 = theAgent.policy_net(input)

        theAgent.Update()
        theAgent.reward_pool = [0]
        theAgent.UpdatePolicy(1)

        action2 = theAgent.policy_net(input).item()
        self.assertNotAlmostEqual(action1, action2, 2)


if (__name__ == '__main__'):
    unittest.main()