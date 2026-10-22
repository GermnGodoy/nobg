import torch

# ---------------------------------------------
# DICE

def dice(pred, target, eps=1e-6):
    """Mean Dice for binary segmentation logits"""
    pred = torch.sigmoid(pred).contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice_score = (2. * intersection + eps) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + eps)
    
    return dice_score.mean()

def dice_loss(pred, target, eps=1e-6):
    return 1 - dice(pred, target, eps)

# ---------------------------------------------
# BCE

def bce(pred, target):
    """Binary Cross Entropy for binary segmentation logits"""
    return torch.nn.functional.binary_cross_entropy_with_logits(pred, target)

def bce_loss(pred, target):
    return bce(pred, target)

# ---------------------------------------------
# Useful Metrics

def false_positive(pred, target):
    """FP / (TP + FP), at probability 0.5"""
    pred = torch.sigmoid(pred) >= 0.5
    target = target >= 0.5
    
    fp = (pred & ~target).sum(dim=2).sum(dim=2)
    total_pred = pred.sum(dim=2).sum(dim=2)
    
    return (fp.float() / (total_pred.float() + 1e-6)).mean()

def true_positive(pred, target):
    """TP / (TP + FN), at probability 0.5"""
    pred = torch.sigmoid(pred) >= 0.5
    target = target >= 0.5
    
    tp = (pred & target).sum(dim=2).sum(dim=2)
    total_target = target.sum(dim=2).sum(dim=2)
    
    return (tp.float() / (total_target.float() + 1e-6)).mean()

# ---------------------------------------------

metrics = {
    "Dice": dice,
    "BCE": bce,
    "False Positive": false_positive,
    "True Positive": true_positive
}
