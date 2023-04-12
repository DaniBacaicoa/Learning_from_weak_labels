import torch
import torch.nn as nn


class PartialLoss2(nn.Module):
    def __init__(self, weak_labels):
        super(PartialLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.weak_labels = weak_labels.detach()
        self.weights = (weak_labels/torch.sum(weak_labels, dim=1, keepdim=True)).detach()


    def forward(self, output, targets, indices):
        #v = output - torch.mean(output, axis=1, keepdims=True)
        p = self.softmax(output)
        L = - torch.sum(self.weights[indices] * targets * logp)

        # Updating used weights in each batch
        new_weights = self.weak_labels[indices] * output.clone().detach()
        #new_weights = self.weak_labels[indices] * torch.exp(logp)
        self.weights[indices] = new_weights/torch.sum(new_weights, dim=1, keepdim=True)
        return L

class PartialLoss(nn.Module):
    def __init__(self, weak_labels):
        super(PartialLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.weak_labels = weak_labels.detach()
        self.weights = (self.weak_labels / torch.sum(self.weak_labels, dim=1, keepdim=True))

    def forward(self,output,targets,indices):
        v = output - torch.mean(output, axis=1, keepdims=True)
        p = self.softmax(v)
        L = -torch.sum(self.weights[indices] * torch.log(p+1e-12))/len(indices)

        revisedY = self.weights[indices].clone()
        revisedY[revisedY > 0] = 1
        revisedY = revisedY * p.clone().detach()
        revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1), 1).transpose(0, 1)

        self.weights[indices] = revisedY

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
        L = - torch.sum(targets * logp)
        return L

class R_CELoss(nn.Module):
    def __init__(self, reg_weight = 0.1, reg_type = 1):
        super(R_CELoss, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.reg_weight = reg_weight
        self.reg_type = reg_type

    def forward(self, inputs, targets, model):
        v = inputs - torch.mean(inputs, axis=1, keepdims=True)
        logp = self.logsoftmax(v)
        L = - torch.sum(targets * logp)
        reg = 0.
        for param in model.parameters():
            reg += torch.norm(param, self.reg_type)
        return L + self.reg_weight * reg

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


class R_LBLoss(nn.Module):
    def __init__(self, k = 1 , beta = 1, reg_weight = 0.1, reg_type = 2):
        super(R_LBLoss, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)
        self.k = k
        self.beta = beta
        self.reg_weight = reg_weight
        self.reg_type = reg_type
    def forward(self, inputs, targets, model):
        v = inputs - torch.mean(inputs, axis=1, keepdims=True)
        logp = self.logsoftmax(v)
        L = - torch.sum(targets * logp) + self.k * torch.sum(torch.abs(v) ** self.beta)
        reg = 0.
        for param in model.parameters():
            reg += torch.norm(param, self.reg_type)
        return L + self.reg_weight * reg

class EMLoss(nn.Module):
    def __init__(self,M):
        super(EMLoss, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.M = M
    def forward(self,out,z):
        logp = logsoftmax(out)
        p = torch.exp(logp)
        Q = p * torch.tensor(self.M[z])
        Q /= torch.sum(Q,dim=1,keepdim=True)
        L = -torch.sum(Q*logp)
        return L

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
