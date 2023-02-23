import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def partial_loss(output,target,true,eps=1e-12):
    out = F.softmax(output)
    l = target * torch.log(out+eps)
    loss = -torch.sum(l)/l.size(0)

    revisedY = target.clone()
    revisedY[revisedY > 0] = 1
    revisedY = revisedY * (out.clone().detach())
    revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)

    return loss, revisedY

class LBLLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LBLLoss, self).__init__()

    def forward(self, inputs, targets, k=1, alpha=1):
        v = inputs - np.mean(inputs, axis=1, keepdims=True)
        logp = self.softmax
        L = - np.sum(targets*logp) + alpha*np.sum(v**2)/2 + k*np.sum(np.abs(v)**beta)


        return L

