
import agentPG

import torch



class AgentPPO(agentPG.AgentPG):
    def Loss(self, actions):
        epsilon = 0.2
        c_1 = 0.05 * 0
        c_2 = 0.0000001 * 0



        v = self.value.GetValue(self.state)
        A = self.discountedReward - v #calc advantage
        
        r = actions / self.policyNetOld(self.state)



        left = r * A
        right = torch.clip(r, 1 - epsilon, 1 + epsilon) * A
        l_1 = torch.min(left, right) #loss
        
        l_2 = self.fMSELoss(v, self.discountedReward)
        
        l_3 = -actions * torch.log2(actions)
        
        loss = l_1 - c_1 * l_2 + c_2 * l_3
        loss = torch.mean(loss) #expected

        print('Total loss for this batch: {}'.format(loss.item()))

        return loss

    def UpdateOld(self):
        self.policyNetOld = self.policyNet

        self.ResetParm()