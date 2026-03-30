import torch.nn as nn

from . import base
from . import functional as F
from ..base.modules import Activation


class JaccardLoss(base.Loss):
    def __init__(self, eps=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )
    
class DiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
       
        y_pr = self.activation(y_pr) #.squeeze(1)

        #print(f"Prediction shape: {y_pr.shape}")  # Debugging
        #print(f"Ground Truth shape: {y_gt.shape}")  # Debugging

        if self.ignore_index is not None:
            mask = (y_gt != self.ignore_index)
            y_pr = y_pr * mask
            y_gt = y_gt * mask

        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass

class DynamicWeightedConfidenceDiceLoss(base.Loss):
    __name__ = "dynamic_weighted_confidence_dice_loss"

    def __init__(self, eps=1., beta=1., amplification_factor=2.0, confidence_threshold=0.8, correctness_threshold=0.5, activation=None, ignore_channels=None, ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.amplification_factor = amplification_factor  # Factor to amplify confident incorrect predictions
        self.confidence_threshold = confidence_threshold  # Confidence threshold for amplification
        self.correctness_threshold = correctness_threshold  # Correctness threshold for amplification
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
        # Apply activation to get probabilities
        y_pr = self.activation(y_pr).squeeze(1)
        
        # Mask out ignore_index pixels if specified
        if self.ignore_index is not None:
            mask = (y_gt != self.ignore_index).float()
            y_pr = y_pr * mask
            y_gt = y_gt * mask

        # Step 1: Calculate confidence as distance from 0.5
        confidence = torch.abs(y_pr - 0.5) * 2  # High values for high confidence

        # Step 2: Calculate correctness as agreement with ground truth
        correctness = torch.abs(y_pr - y_gt.float())  # High values for incorrect predictions

        # Step 3: Calculate initial weights
        weights = 1 - confidence * (1 - correctness)

        # Step 4: Amplify weights for confident, incorrect predictions
        weights = torch.where(
            (confidence > self.confidence_threshold) & (correctness > self.correctness_threshold),
            weights * self.amplification_factor,
            weights
        )

        # Normalize weights to avoid instability
        weights = weights / (weights.mean() + 1e-8)

        # Calculate the weighted Dice loss
        intersection = (weights * y_pr * y_gt).sum()
        denominator = (weights * (y_pr + y_gt)).sum()
        dice_loss = 1 - (2 * intersection + self.eps) / (denominator + self.eps)
        
        return dice_loss