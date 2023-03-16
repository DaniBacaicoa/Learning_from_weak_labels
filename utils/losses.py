import torch
import torch.nn as nn


class PartialLoss(nn.Module):
    def __init__(self, weak_labels):
        super(PartialLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax()
        self.weak_labels = weak_labels
        self.weights = weak_labels/torch.sum(weak_labels, dim=1, keepdim=True)


    def forward(self, output, targets, indices):
        v = output - torch.mean(output, axis=1, keepdims=True)
        logp = self.logsoftmax(v)
        L = - torch.sum(self.weights[indices].detach() * targets * logp)

        # Updating used weights in each batch
        #new_weights = self.weak_labels[indices].detach() * output.clone().detach()
        new_weights = self.weak_labels[indices].detach() * torch.exp(logp).clone().detach()
        self.weights[indices] = new_weights/torch.sum(new_weights, dim=1, keepdim=True)
        return L

'''
tbd. try to use this as a counterpart to hardmax
class GumbelLoss(nn.Module):
    #this tries to soften the constraint imposed by the non-differentiability of the minimum
    def __init__(self,tau = 0.1:
        super(GumbelLoss, self).__init__()
        self.tau = tau

    def forward(self, output, targets):
'''



class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        v = inputs - torch.mean(inputs, axis=1, keepdims=True)
        logp = self.logsoftmax(v)
        L = - torch.sum(targets*logp)
        return L

class BrierLoss(nn.Module):
    def __init__(self):
        super(BrierLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, inputs, targets):
        v = inputs# - torch.mean(inputs, axis=1, keepdims=True)
        p = self.softmax(v)
        L = torch.sum((targets - p)**2)
        return L

class LBLoss(nn.Module):
    def __init__(self, k=1, beta=1):
        super(LBLoss, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)
        self.k = k
        self.beta = beta


    def forward(self, inputs, targets):
        v = inputs - torch.mean(inputs, axis=1, keepdims=True)
        logp = self.logsoftmax(v)
        L = - torch.sum(targets * logp) + self.k * torch.sum(torch.abs(v) ** self.beta)
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

class OSLCELoss(nn.Module):
    def __init__(self):
        super(OSLCELoss, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)

    def hardmax(self, A):
        D = torch.eq(A, torch.max(A, axis=1, keepdims=True)[0])
        return D / torch.sum(D, axis=1, keepdims=True)

    def forward(self, inputs, targets):
        logp = self.logsoftmax(inputs)
        p = torch.exp(logp)
        D = self.hardmax(targets * p)
        L = - torch.sum(D*logp)
        return L

class OSLBrierLoss(nn.Module):
    def __init__(self):
        super(OSLBrierLoss, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)

    def hardmax(self, A):
        D = torch.eq(A, torch.max(A, axis=1, keepdims=True)[0])
        return D / torch.sum(D, axis=1, keepdims=True)

    def forward(self, inputs, targets):
        p = torch.exp(self.logsoftmax(inputs))
        D = self.hardmax(targets * p)
        L = torch.sum((D - p)**2)/2
        return L
