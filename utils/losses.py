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


class CELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CELoss, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        v = inputs - torch.mean(inputs, axis=1, keepdims=True)
        logp = self.logsoftmax(v)
        L = - torch.sum(targets*logp)
        return L

class BrierLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BrierLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, inputs, targets):
        v = inputs# - torch.mean(inputs, axis=1, keepdims=True)
        p = self.softmax(v)
        L = torch.sum((targets - p)**2)
        return L

class LBLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LBLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, inputs, targets, k=1, beta=1):
        v = inputs - torch.mean(inputs, axis=1, keepdims=True)
        logp = self.softmax(v)
        L = - torch.sum(targets * logp) + k * torch.sum(torch.abs(v) ** beta)
        return L

class EMLoss(nn.Module):
    def __init__(self,M):
        super(EMLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim = 1)
        self.M = M

    def forward(self,inputs,targets):
        T = torch.max(targets,dim=1)
        v = inputs  # - torch.mean(inputs, axis=1, keepdims=True)
        p = self.softmax(v)
        Q = p * M[tor]