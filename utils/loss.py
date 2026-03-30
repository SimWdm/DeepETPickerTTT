#%%
from torch import nn as nn
import torch
import torch.nn.functional as F

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


# Dice loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, args=None, beta=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.use_softmax = args.use_softmax
        self.use_sigmoid = args.use_sigmoid
        self.beta = beta


    # def forward(self, outputs, targets, trust_labels=True, beta=1):
    #     if not trust_labels.all().item():
    #         raise NotImplementedError("Un-trusted labels not implemented in this version.")
    #     if not targets.shape[-1] == targets.shape[-2] == targets.shape[-3]:
    #         raise ValueError("Target tensor must be cubic, in spatial dimensions.")
    #     # flatten label and prediction tensors
    #     outputs = flatten(outputs)
    #     targets = flatten(targets)

    #     intersection = (outputs * targets).sum(-1)
    #     dice = (2. * intersection + self.smooth) / (outputs.sum(-1) + targets.sum(-1) + self.smooth)
    #     return 1 - dice.mean()
        
    def forward(self, outputs, targets, trust_labels=True):
        if not trust_labels.all().item():
            raise NotImplementedError("Un-trusted labels not implemented in this version.")
        if not targets.shape[-1] == targets.shape[-2] == targets.shape[-3]:
            raise ValueError("Target tensor must be cubic, in spatial dimensions.")
        # flatten label and prediction tensors
        outputs = flatten(outputs)
        targets = flatten(targets)

        tp = (outputs * targets).sum(-1)
        fn = ((1 - outputs) * targets).sum(-1)
        fp = (outputs * (1 - targets)).sum(-1)
        dice = ((1+self.beta**2) * tp + self.smooth) / ((tp + fn) * self.beta**2 + (tp + fp) + self.smooth)
        return 1 - dice.mean()
        
    
class CELoss(nn.Module):
    def __init__(self, args=None):
        super(CELoss, self).__init__()
        self.args = args
        
    def standard_ce_loss(self, outputs, targets, weights=None):
        outputs = outputs.clamp(min=1e-8)  # to avoid NaNs
        targets_idx = targets.argmax(dim=1).long()  # (N, D, H, W)    
    
        loss_ce = F.nll_loss(outputs.log(), targets_idx, reduction='none', weight=weights) # (N, D, H, W)
        loss_ce = loss_ce.mean(dim=(-1, -2, -3))  # (N,)
        return loss_ce.mean()

    def forward(self, outputs, targets, trust_labels=None): 
        return self.standard_ce_loss(outputs, targets)
    

class BCELoss(nn.Module):
    def __init__(self, args=None):
        super(BCELoss, self).__init__()
        self.args = args

    def standard_bce_loss(self, outputs, targets, weights=None):
        """
        outputs: (N, 1, D, H, W) or (N, D, H, W) — model predictions in [0, 1]
        targets: same shape, ground truth labels in {0, 1}
        weights: optional tensor of per-class or per-voxel weights
        """
        # clamp to avoid log(0)
        outputs = outputs.clamp(min=1e-8, max=1 - 1e-8)
        
        # compute BCE per voxel
        loss_bce = F.binary_cross_entropy(outputs, targets, reduction='none', weight=weights)
        
        # mean over spatial dimensions
        loss_bce = loss_bce.mean(dim=(-1, -2, -3))  # (N,)
        
        # mean across batch
        return loss_bce.mean()

    def forward(self, outputs, targets, trust_labels=None):
        return self.standard_bce_loss(outputs, targets)
    