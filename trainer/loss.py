import numpy as np
import torch

# ---------------------------------------------
# DICE

def dice(pred, target, eps=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice_score = (2. * intersection + eps) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + eps)
    
    return dice_score.mean()

def dice_loss(pred, target, eps=1e-6):
    return 1 - dice(pred, target, eps)

# ---------------------------------------------
# Useful Metrics

def false_positive(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()
    
    fp = ((pred == 1) & (target == 0)).sum(dim=2).sum(dim=2)
    total_pred = (pred == 1).sum(dim=2).sum(dim=2)
    
    return (fp.float() / (total_pred.float() + 1e-6)).mean()

def true_positive(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()
    
    tp = ((pred == 1) & (target == 1)).sum(dim=2).sum(dim=2)
    total_target = (target == 1).sum(dim=2).sum(dim=2)
    
    return (tp.float() / (total_target.float() + 1e-6)).mean()