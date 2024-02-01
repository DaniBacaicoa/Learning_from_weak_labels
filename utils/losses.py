import torch
import torch.nn as nn




class CELoss(nn.Module):
    """
    Cross entropy loss. calculates the discrepancy between the prediction and the actual label. 
    loss = -y log(f)

    Args:
        None

    Returns:
        - torch.Tensor: The computed cross-entropy loss

    Example:
        >>> loss_function = Cross_entropy()
        >>> inputs = torch.tensor([[1.2, 0.5, -0.8],
                                   [0.1, 0.3, 0.6]])
        >>> targets = torch.tensor([[0, 1, 0],
                                    [1, 0, 0]])
        >>> loss = loss_function(inputs, targets)
        >>> print(loss)
        tensor(1.2096)
    """
    def __init__(self):
        super(CELoss, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)

    def forward(self, inputs, targets):
        """
        Compute the cross-entropy loss.

        Args:
            inputs (torch.Tensor): Predicted logits.
            targets (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: The computed cross-entropy loss.
        """
        v = inputs - torch.mean(inputs, axis = 1, keepdims = True)
        logp = self.logsoftmax(v)
        L = - torch.sum(targets * logp)
        return L

class BrierLoss(nn.Module):
    """
    Brier loss. Calculates the mean squared error between the predicted probabilities and the actual label.

    Args:
        None

    Returns:
        - torch.Tensor: The computed Brier loss

    Example:
        >>> loss_function = BrierLoss()
        >>> inputs = torch.tensor([[0.9, 0.05, 0.05],
                                   [0.2, 0.3, 0.5]])
        >>> targets = torch.tensor([[1, 0, 0],
                                    [0, 1, 0]])
        >>> loss = loss_function(inputs, targets)
        >>> print(loss)
        tensor(0.0675)
    """
    def __init__():
        super(BrierLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, inputs, targets):
        """
        Compute the Brier loss.

        Args:
            inputs (torch.Tensor): Predicted probabilities.
            targets (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: The computed Brier loss.
        """
        v = inputs - torch.mean(inputs, axis = 1, keepdims = True)
        p = self.softmax(v)
        L = torch.sum((targets - p)**2)
        return L




'''
class PartialLoss2(nn.Module):
    def __init__(self, weak_labels):
        super(PartialLoss2, self).__init__()
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
'''

'''
class PartialLoss(nn.Module):
    def __init__(self, weak_labels):
        super(PartialLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        #self.weak_labels = weak_labels / torch.sum(weak_labels, dim=1, keepdim=True)
        self.weights = nn.Parameter(torch.Tensor(weak_labels / torch.sum(weak_labels, dim=1, keepdim=True)),requires_grad=False)

    def forward(self, output, targets, indices):
        #print(self.weights)
        '''
        # targets are not used as we use the updated weights.
        # anyway they are left there to maintain coherence with the rest of the losses
        #v = output - torch.mean(output, axis=1, keepdims=True)
        '''
        # We take the weights in this batch
        weight = self.weights[indices]
        weight = weight.detach()

        logp = self.logsoftmax(output)
        L = - torch.sum(weight * logp) / len(indices)

        weight = torch.where(weight > 0, torch.tensor(1, dtype=weight.dtype, device=weight.device), weight)
        weight = weight * torch.exp(logp).clone().detach()
        weight = weight / torch.sum(weight, dim=1, keepdim=True)

        self.weights[indices] = weight

        return L


class PartialLoss_b(nn.Module):
    def __init__(self, weak_labels):
        super(PartialLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.weak_labels = weak_labels.clone().detach()
        self.weights = (self.weak_labels / torch.sum(self.weak_labels, dim=1, keepdim=True))

    def forward(self,output,targets,indices):
        '''
        # targets are not used as we use the updated weights.
        # anyway they are left there to maintain coherence with the rest of the losses
        #v = output - torch.mean(output, axis=1, keepdims=True)
        '''
        logp = self.logsoftmax(output)
        L = - torch.sum(self.weights[indices] * logp)/len(indices)


        revisedY = self.weights[indices].clone()
        revisedY[revisedY > 0] = 1
        revisedY = revisedY * torch.exp(logp).clone().detach()
        revisedY = revisedY / torch.sum(revisedY, dim=1, keepdim=True)

        self.weights[indices] = revisedY

        return L


class PartialLoss_b2(nn.Module):
    def __init__(self, weak_labels):
        super(PartialLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.weak_labels = weak_labels.clone().detach()
        self.weights = (self.weak_labels / torch.sum(self.weak_labels, dim=1, keepdim=True))

    def forward(self,output,targets,indices):
        '''
        # targets are not used as we use the updated weights.
        # anyway they are left there to maintain coherence with the rest of the losses
        #v = output - torch.mean(output, axis=1, keepdims=True)
        '''
        logp = self.logsoftmax(output)
        L = - torch.sum(self.weights[indices] * logp)/len(indices)


        revisedY = self.weights[indices].clone()
        revisedY[revisedY > 0] = 1
        revisedY = revisedY * torch.exp(logp).clone().detach()
        revisedY = revisedY / torch.sum(revisedY, dim=1, keepdim=True)

        self.weights[indices] = revisedY

        return L


def partial_loss_b3(out, targ, true, eps=1e-12):
    softmax = nn.Softmax(dim=1)
    p = softmax(out)
    l = targ * torch.log(p+eps)
    loss = -torch.sum(l)/l.size(0)

    revisedY = targ.clone()
    revisedY[revisedY > 0] = 1
    revisedY = revisedY * (p.clone().detach())
    revisedY = revisedY / torch.sum(revisedY, dim=1, keepdim=True)
    new = revisedY

    return loss, new




tbd. try to use this as a counterpart to hardmax
class GumbelLoss(nn.Module):
    #this tries to soften the constraint imposed by the non-differentiability of the minimum
    def __init__(self,tau = 0.1:
        super(GumbelLoss, self).__init__()
        self.tau = tau

    def forward(self, output, targets):



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
        logp = self.logsoftmax(out)

        p = torch.exp(logp)
        Q = p.detach() * torch.tensor(self.M[z])
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


class EMLoss_new(nn.Module):
    def __init__(self, n_weak, n_true):
        super(EMLoss_new, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.n_true = n_true
        self.n_weak = n_weak
        self.M = nn.Parameter(torch.ones((n_weak,n_true)) / n_weak, requires_grad=True)

    def forward(self, out, z):
        logp = self.logsoftmax(out)
        p = torch.exp(logp)
        Q = self.est_true_label_dist(p, z)
        self.update_noisy_prior(Q, z)
        loss = -torch.sum(Q * logp)
        return loss

    def est_true_label_dist(self, p, z):
        Q = p * self.M[z]
        Q /= torch.sum(Q, dim=1, keepdim=True)
        return Q

    def update_noisy_prior(self, Q, z):
        M_new = torch.zeros_like(self.M)
        for i in range(self.n_true):
            if i in z:
                print(Q.shape,Q[1:10,:])
                print(z.shape,z)

                print(Q[z==i,:])

                print(torch.sum(Q[z==i,:],dim=1))
                QQ = torch.mean(Q[z==i,:],dim=1)/torch.sum(torch.mean(Q[z==i,:],dim=1))
                #print(QQ.shape, QQ[1:10, :])
                print(M_new.shape,M_new[:,i].shape,M_new)
                print(QQ.shape,QQ)

                M_new[:,i] = QQ
            else:
                M_new[:,i] = self.M[:,i]
        self.M.data = M_new

        # instead of having the M, with the Q could we have a instance dependent?



class EMLoss2(nn.Module):
    def __init__(self, d, c, M_est ):
        super(EMLoss2, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

        self.M_hat = nn.Parameter(M_est, requires_grad= True)

    def forward(self, out, z):
        logp = self.logsoftmax(out)
        p = torch.exp(logp)

        Q = p.detach() * self.M_hat[z]
        Q /= torch.sum(Q, dim=1, keepdim=True)

        L = -torch.sum(Q * logp)

        return L

'''