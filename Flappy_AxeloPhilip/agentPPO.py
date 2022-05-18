
import agentPG

import torch



class AgentPPO(agentPG.AgentPG):
    def Loss(self, actions):
        epsilon = 0.2
        c_1 = 0
        c_2 = 0

        A = self.discountedReward - self.value.GetValue(self.state) #calc advantage
        
        r = actions / self.policyNetOld(self.state)

        left = r * A
        right = torch.clip(r, 1 - epsilon, 1 + epsilon) * A
        
        l_1 = torch.min(left, right) #loss
        l_1 = torch.mean(l_1) #expected
        
        
        
        l_2 = self.fMSELoss(self.state, self.v_targ_tensor)
        l_2 = torch.mean(l_2) #expected



        l_3 = -actions * torch.log2(actions)
        l_3 = torch.mean(l_3) #expected



        loss = l_1 - c_1 * l_2 + c_2 * l_3

        print('Total loss for this batch: {}'.format(loss.item()))

        return loss

    def UpdateOld(self):
        self.policyNetOld = self.policyNet