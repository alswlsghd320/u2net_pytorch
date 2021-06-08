import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO: focal loss index over error
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def multi_focal_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    focal_loss = FocalLoss(size_average=True)

    loss0 = focal_loss(d0, labels_v)
    loss1 = focal_loss(d1, labels_v)
    loss2 = focal_loss(d2, labels_v)
    loss3 = focal_loss(d3, labels_v)
    loss4 = focal_loss(d4, labels_v)
    loss5 = focal_loss(d5, labels_v)
    loss6 = focal_loss(d6, labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    #print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss0, loss

def multi_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    bce_loss = nn.BCELoss(size_average=True)

    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    #print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss0, loss

# def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
#     """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
#     Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
#     where Loss is one of the standard losses used for Neural Networks.
#     Args:
#       labels: A int tensor of size [batch].
#       logits: A float tensor of size [batch, no_of_classes].
#       samples_per_cls: A python list of size [no_of_classes].
#       no_of_classes: total number of classes. int
#       loss_type: string. One of "sigmoid", "focal", "softmax".
#       beta: float. Hyperparameter for Class balanced loss.
#       gamma: float. Hyperparameter for Focal loss.
#     Returns:
#       cb_loss: A float tensor representing class balanced loss
#     """
#     effective_num = 1.0 - np.power(beta, samples_per_cls)
#     weights = (1.0 - beta) / np.array(effective_num)
#     weights = weights / np.sum(weights) * no_of_classes
#
#     labels_one_hot = F.one_hot(labels, no_of_classes).float()
#
#     weights = torch.tensor(weights).float()
#     weights = weights.unsqueeze(0)
#     weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
#     weights = weights.sum(1)
#     weights = weights.unsqueeze(1)
#     weights = weights.repeat(1,no_of_classes)
#
#     if loss_type == "focal":
#         cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
#     elif loss_type == "sigmoid":
#         cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
#     elif loss_type == "softmax":
#         pred = logits.softmax(dim = 1)
#         cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
#     return cb_loss