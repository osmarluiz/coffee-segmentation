"""
Mask2Former training utilities.

Custom Epoch subclasses that:
  * receive the rich dict output of the Mask2Former model,
  * build per-image GT instance targets ({'labels', 'masks'}) from a dense
    (B, H, W) integer mask,
  * feed the rich predictions and the instance targets to the
    Hungarian-matched ``SetCriterion``,
  * return the semantic ``(B, C, H, W)`` tensor for downstream metrics
    (mIoU is argmax-based so it works seamlessly).

These mirror ``smp.utils.train.TrainEpoch`` / ``ValidEpoch`` but with a
batch_update that knows about the set-prediction interface.
"""

import contextlib
from typing import List, Optional

import torch

from segmentation_models_pytorch.utils.train import Epoch, _autocast_ctx


def build_m2f_targets(masks: torch.Tensor) -> List[dict]:
    """Convert dense (B, H, W) integer masks into per-image instance targets.

    For semantic segmentation, each class present in the image becomes one
    "instance" with a binary mask of all pixels of that class. Images with
    only a single class still produce one target so the Hungarian matcher
    has something to match.

    Args:
        masks: (B, H, W) integer tensor with values in ``{0, ..., num_classes-1}``.

    Returns:
        List of B dicts:
          - ``labels``: (N_b,) long tensor of present class ids in the image
          - ``masks``:  (N_b, H, W) float tensor of binary GT masks
    """
    targets = []
    for b in range(masks.shape[0]):
        m = masks[b]  # (H, W)
        present = torch.unique(m).long()
        instance_masks = torch.stack([(m == c).float() for c in present], dim=0)
        targets.append({"labels": present, "masks": instance_masks})
    return targets


class Mask2FormerTrainEpoch(Epoch):
    """Training epoch using SetCriterion + Hungarian matcher."""

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        device: str = "cpu",
        verbose: bool = True,
        amp_dtype: Optional[torch.dtype] = None,
        max_grad_norm: float = 0.01,
    ):
        super().__init__(
            model=model,
            loss=criterion,           # criterion is treated as the "loss" object
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
            amp_dtype=amp_dtype,
        )
        self.optimizer = optimizer
        # Aggressive gradient clipping matches the official Mask2Former recipe
        # and stops the early-iteration loss spike we see from random init +
        # AdamW.
        self.max_grad_norm = max_grad_norm

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        with _autocast_ctx(self.amp_dtype, self.device):
            output = self.model(x)
        # Build per-image instance targets BEFORE the autocast cast so the GT
        # stays in fp32; mask CE/dice losses are stable that way.
        targets = build_m2f_targets(y)
        loss = self.loss(output, targets)
        loss.backward()
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )
        self.optimizer.step()
        # Return semantic tensor in fp32 for metrics (mIoU does argmax)
        semantic = output["semantic"].float().detach()
        return loss, semantic


class Mask2FormerValidEpoch(Epoch):
    """Validation epoch using the same Hungarian criterion.

    The BN-in-train-mode hack required for early-epoch validation is baked
    into ``Mask2Former.eval()`` itself, so calling ``self.model.eval()`` here
    is enough.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        device: str = "cpu",
        verbose: bool = True,
        amp_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            model=model,
            loss=criterion,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
            amp_dtype=amp_dtype,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            with _autocast_ctx(self.amp_dtype, self.device):
                output = self.model(x)
            targets = build_m2f_targets(y)
            loss = self.loss(output, targets)
            semantic = output["semantic"].float()
        return loss, semantic
