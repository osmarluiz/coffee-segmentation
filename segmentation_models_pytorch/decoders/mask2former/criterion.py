"""
Mask2Former set-prediction criterion.

Given the bipartite matching produced by HungarianMatcher, compute:
  * loss_ce:   classification cross-entropy on every query, where unmatched
               queries are assigned the "no object" class (index = num_classes).
               The no-object class is down-weighted by ``eos_coef``.
  * loss_mask: sigmoid binary cross-entropy on the masks of MATCHED queries
               only, against the GT instance masks (downsampled to the mask
               prediction resolution).
  * loss_dice: dice loss on the same matched pairs.

Per the paper, the same losses are applied to every intermediate decoder layer
(deep supervision); we use ``aux_outputs`` from the model for that.

Reference: Cheng et al., "Masked-attention Mask Transformer for Universal
Image Segmentation", CVPR 2022.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: int) -> torch.Tensor:
    """Per-pixel BCE, averaged over pixels then summed over masks and divided by num_masks."""
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / max(num_masks, 1)


def _dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: int) -> torch.Tensor:
    """Multiclass dice loss summed over masks and divided by num_masks."""
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2.0 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1.0 - (numerator + 1.0) / (denominator + 1.0)
    return loss.sum() / max(num_masks, 1)


class SetCriterion(nn.Module):
    """Hungarian-matched set criterion for Mask2Former."""

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1,
        aux_loss: bool = True,
        class_weight: torch.Tensor = None,
    ):
        """
        Args:
            num_classes: number of real classes (the no-object class will be
                added internally as index ``num_classes``).
            matcher: a HungarianMatcher instance.
            weight_dict: mapping from loss name to scalar weight. Keys are
                ``'loss_ce'``, ``'loss_mask'``, ``'loss_dice'``.
            eos_coef: down-weighting factor for the no-object class in the
                classification CE.
            aux_loss: if True, apply the same losses to ``outputs['aux_outputs']``
                for deep supervision.
            class_weight: optional (num_classes,) tensor of per-real-class
                weights to combine with eos_coef. Used to incorporate the
                [1, 15, 35] class imbalance scheme of the coffee paper.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.aux_loss = aux_loss

        # Build empty_weight = [class_weights..., eos_coef]
        if class_weight is None:
            class_weight = torch.ones(num_classes)
        else:
            class_weight = torch.as_tensor(class_weight, dtype=torch.float32)
            if class_weight.numel() != num_classes:
                raise ValueError(
                    f"class_weight must have {num_classes} entries, got "
                    f"{class_weight.numel()}"
                )
        empty_weight = torch.cat([class_weight, torch.tensor([eos_coef])])
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, b) for b, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices,
    ) -> Dict[str, torch.Tensor]:
        src_logits = outputs["pred_logits"]  # (B, Q, C+1)
        B, Q, _ = src_logits.shape

        target_classes = torch.full(
            (B, Q),
            self.num_classes,  # no-object index
            dtype=torch.long,
            device=src_logits.device,
        )
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() > 0:
                target_classes[b, src_idx.to(src_logits.device)] = (
                    targets[b]["labels"][tgt_idx.to(src_logits.device)].long()
                )

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2).float(),
            target_classes,
            weight=self.empty_weight.to(src_logits.device),
        )
        return {"loss_ce": loss_ce}

    def loss_masks(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices,
    ) -> Dict[str, torch.Tensor]:
        src_masks = outputs["pred_masks"]  # (B, Q, h, w)
        mh, mw = src_masks.shape[-2:]

        src_collected, tgt_collected = [], []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_collected.append(
                src_masks[b][src_idx.to(src_masks.device)]
            )  # (n_b, h, w)
            tgt = targets[b]["masks"][tgt_idx.to(src_masks.device)].float()  # (n_b, H, W)
            tgt = F.interpolate(
                tgt.unsqueeze(1), size=(mh, mw), mode="nearest"
            ).squeeze(1)  # (n_b, h, w)
            tgt_collected.append(tgt)

        if not src_collected:
            zero = src_masks.sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}

        src_cat = torch.cat(src_collected, dim=0).float()  # (M, h, w)
        tgt_cat = torch.cat(tgt_collected, dim=0)          # (M, h, w)
        M = src_cat.shape[0]

        src_flat = src_cat.flatten(1)
        tgt_flat = tgt_cat.flatten(1)

        return {
            "loss_mask": _sigmoid_ce_loss(src_flat, tgt_flat, M),
            "loss_dice": _dice_loss(src_cat, tgt_cat, M),
        }

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute the total weighted loss across the final layer + aux layers."""
        main_keys = {"pred_logits", "pred_masks"}
        main_outputs = {k: outputs[k] for k in main_keys}

        # Final layer
        indices = self.matcher(main_outputs, targets)
        losses = {}
        losses.update(self.loss_labels(main_outputs, targets, indices))
        losses.update(self.loss_masks(main_outputs, targets, indices))

        # Auxiliary layers (deep supervision)
        if self.aux_loss and "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_main = {k: aux[k] for k in main_keys}
                aux_indices = self.matcher(aux_main, targets)
                aux_l = {}
                aux_l.update(self.loss_labels(aux_main, targets, aux_indices))
                aux_l.update(self.loss_masks(aux_main, targets, aux_indices))
                for k, v in aux_l.items():
                    losses[f"{k}_aux_{i}"] = v

        total = src_logits = None  # only for the "no losses" edge case
        total = torch.zeros((), device=outputs["pred_logits"].device)
        for k, v in losses.items():
            base = k.split("_aux_")[0]
            w = self.weight_dict.get(base, 0.0)
            total = total + w * v

        # Stash component losses for logging (no grad needed)
        self.last_losses = {k: v.detach() for k, v in losses.items()}
        return total

    @property
    def __name__(self) -> str:  # for AverageValueMeter logging
        return "m2f_set_loss"
