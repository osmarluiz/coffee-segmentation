"""
Custom loss functions for semantic segmentation.
Multiclass implementations of Dice Loss and Focal Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassDiceLoss(nn.Module):
    """
    Multiclass Dice Loss.

    Computes Dice coefficient for each class and averages.
    Effective for imbalanced datasets by optimizing overlap directly.

    Args:
        num_classes: Number of classes (default: 3)
        weight: Per-class weights (default: None for equal weighting)
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
    """

    def __init__(self, num_classes=3, weight=None, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.weight = torch.FloatTensor(weight) if weight else torch.ones(num_classes)

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        weight = self.weight.to(pred.device)

        dice_scores = []
        for c in range(self.num_classes):
            pred_c = pred[:, c]
            target_c = target_one_hot[:, c]
            intersection = (pred_c * target_c).sum(dim=(1, 2))
            dice = (2.0 * intersection + self.smooth) / (
                pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2)) + self.smooth)
            dice_scores.append(dice * weight[c])

        dice_scores = torch.stack(dice_scores, dim=1)
        mean_dice = dice_scores.sum() / (self.num_classes * pred.shape[0])
        return 1.0 - mean_dice


class MulticlassFocalLoss(nn.Module):
    """
    Multiclass Focal Loss.

    Down-weights easy examples to focus on hard ones.
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)

    Args:
        num_classes: Number of classes (default: 3)
        alpha: Per-class weights (default: None)
        gamma: Focusing parameter, higher = more focus on hard examples (default: 2.0)
        reduction: 'mean' or 'sum' (default: 'mean')
    """

    def __init__(self, num_classes=3, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = torch.FloatTensor(alpha) if alpha else torch.ones(num_classes)

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=1)
        probs = torch.exp(log_probs)

        B, C, H, W = pred.shape
        log_probs = log_probs.permute(0, 2, 3, 1).contiguous().view(-1, C)
        probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)
        target = target.view(-1)

        alpha = self.alpha.to(pred.device)
        target_one_hot = F.one_hot(target, self.num_classes).float()
        pt = (probs * target_one_hot).sum(dim=1)
        log_pt = (log_probs * target_one_hot).sum(dim=1)

        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = alpha[target]
        focal_loss = -alpha_weight * focal_weight * log_pt

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CombinedLoss(nn.Module):
    """Combined Dice + Focal Loss."""

    def __init__(self, num_classes=3, alpha_dice=0.5, alpha_focal=0.5,
                 dice_weight=None, focal_alpha=None, focal_gamma=2.0):
        super().__init__()
        self.alpha_dice = alpha_dice
        self.alpha_focal = alpha_focal
        self.dice_loss = MulticlassDiceLoss(num_classes, weight=dice_weight)
        self.focal_loss = MulticlassFocalLoss(num_classes, alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, pred, target):
        return self.alpha_dice * self.dice_loss(pred, target) + \
               self.alpha_focal * self.focal_loss(pred, target)
