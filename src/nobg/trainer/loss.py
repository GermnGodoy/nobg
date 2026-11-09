import torch
from torch.nn import functional as F

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
# Matting

def mae(pred, target):
    """Mean absolute error of continuous alpha, from logits."""
    return F.l1_loss(pred.sigmoid(), target)


def mse(pred, target):
    """Mean squared error of continuous alpha, from logits."""
    return F.mse_loss(pred.sigmoid(), target)


def l1_loss(pred, target):
    return mae(pred, target)


def gradient_loss(pred, target):
    """L1 error of horizontal and vertical differences in continuous alpha."""
    alpha = pred.sigmoid()
    loss = alpha.mean() * 0
    for dim in (2, 3):
        if alpha.shape[dim] > 1:
            loss = loss + F.l1_loss(
                torch.diff(alpha, dim=dim), torch.diff(target, dim=dim),
            )
    return loss


def detail_loss(pred, target, kernel_size=7):
    """Continuous-alpha MAE on transitions and translucent regions, per image.

    Normalize by the detail area so thin hair and soft edges cannot be diluted
    by large regions of entirely opaque foreground or empty background.
    """
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Detail kernel_size must be a positive odd integer")
    padding = kernel_size // 2
    maximum = F.max_pool2d(target, kernel_size, stride=1, padding=padding)
    minimum = -F.max_pool2d(-target, kernel_size, stride=1, padding=padding)
    region = (((target > 0) & (target < 1)) | (maximum - minimum > 1e-6)).to(target.dtype)
    error = (pred.sigmoid() - target).abs() * region
    dims = (1, 2, 3)
    return (error.sum(dims) / region.sum(dims).clamp_min(1)).mean()

# ---------------------------------------------
# Edges

def get_edges(target, kernel_size=7):
    """Binary morphological gradient; pooling ignores pixels outside the image."""
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Boundary kernel_size must be a positive odd integer")
    
    binary = (target >= 0.5).to(target.dtype)
    padding = kernel_size // 2
    dilated = F.max_pool2d(binary, kernel_size, stride=1, padding=padding)
    other = -F.max_pool2d(-binary, kernel_size, stride=1, padding=padding)
    return dilated - other


def edge_loss(pred, target, kernel_size=7):
    edges = get_edges(target, kernel_size)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

    return (loss * edges).sum() / edges.sum().clamp_min(1)


def edge_f1(pred, target):
    """Mean contour F1 at probability 0.5, with a two-pixel matching tolerance"""
    
    pred_edges = get_edges((pred >= 0).to(target.dtype), kernel_size=3)
    target_edges = get_edges(target, kernel_size=3)
    
    pred_dilated = F.max_pool2d(pred_edges, 5, stride=1, padding=2)
    target_dilated = F.max_pool2d(target_edges, 5, stride=1, padding=2)
    
    dims = (1, 2, 3)
    
    pred_count, target_count = pred_edges.sum(dims), target_edges.sum(dims)
    precision = (pred_edges * target_dilated).sum(dims) / pred_count.clamp_min(1)
    recall = (target_edges * pred_dilated).sum(dims) / target_count.clamp_min(1)
    
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
    empty = (pred_count == 0) & (target_count == 0)
    
    return torch.where(empty, torch.ones_like(f1), f1).mean()

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
    "MAE": mae,
    "MSE": mse,
    "Detail MAE": detail_loss,
    "Dice": dice,
    "BCE": bce,
    "Boundary F1": edge_f1,
    "False Positive": false_positive,
    "True Positive": true_positive
}
