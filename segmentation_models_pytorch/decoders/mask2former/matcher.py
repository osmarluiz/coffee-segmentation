"""
Hungarian bipartite matcher for Mask2Former.

Given the model's per-query predictions (class logits + mask logits) and the
ground-truth instances (one per present class for semantic segmentation), this
module finds the optimal one-to-one assignment between queries and GT
instances that minimises the matching cost.

Match cost is a weighted sum of:
  * a negative-class-probability term (Cheng et al. 2022, Sec. 3.2),
  * a sigmoid binary cross-entropy term on the predicted mask,
  * a dice term on the predicted mask.

Reference: Cheng et al., "Masked-attention Mask Transformer for Universal
Image Segmentation", CVPR 2022.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as e:
    raise ImportError(
        "Mask2Former Hungarian matching requires scipy. "
        "Install with `pip install scipy`."
    ) from e


@torch.no_grad()
def _batch_dice_cost(pred_masks: torch.Tensor, tgt_masks: torch.Tensor) -> torch.Tensor:
    """Pairwise dice cost between Q predicted masks and N GT masks.

    Args:
        pred_masks: (Q, HW) sigmoided predictions, flattened.
        tgt_masks:  (N, HW) binary targets, flattened.

    Returns:
        (Q, N) cost matrix; lower is better.
    """
    numerator = 2 * torch.einsum("qc,nc->qn", pred_masks, tgt_masks)
    denominator = pred_masks.sum(-1)[:, None] + tgt_masks.sum(-1)[None, :]
    return 1.0 - (numerator + 1.0) / (denominator + 1.0)


@torch.no_grad()
def _batch_sigmoid_ce_cost(pred_logits: torch.Tensor, tgt_masks: torch.Tensor) -> torch.Tensor:
    """Pairwise BCE-with-logits cost.

    Args:
        pred_logits: (Q, HW) raw mask logits.
        tgt_masks:   (N, HW) binary targets, flattened.

    Returns:
        (Q, N) cost matrix; lower is better.
    """
    hw = pred_logits.shape[-1]
    pos = F.binary_cross_entropy_with_logits(
        pred_logits, torch.ones_like(pred_logits), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        pred_logits, torch.zeros_like(pred_logits), reduction="none"
    )
    loss = torch.einsum("qc,nc->qn", pos, tgt_masks) + torch.einsum(
        "qc,nc->qn", neg, 1.0 - tgt_masks
    )
    return loss / hw


class HungarianMatcher(nn.Module):
    """Bipartite Hungarian matcher between Mask2Former queries and GT instances."""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            outputs: dict with
                'pred_logits': (B, Q, C+1)
                'pred_masks':  (B, Q, h, w)
            targets: list of B dicts, each with
                'labels': (N_b,) long tensor of present class ids
                'masks':  (N_b, H, W) binary tensor at full resolution

        Returns:
            List of B tuples (src_idx, tgt_idx). src_idx[i] is the query index
            in [0, Q) matched to tgt_idx[i], the GT-instance index in [0, N_b).
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_pred_masks = outputs["pred_masks"]  # (B, Q, h, w)
        mh, mw = out_pred_masks.shape[-2:]

        indices = []
        for b in range(bs):
            tgt_labels = targets[b]["labels"]  # (N_b,)
            tgt_masks_full = targets[b]["masks"]  # (N_b, H, W)
            n_targets = tgt_labels.shape[0]
            if n_targets == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.long),
                        torch.empty(0, dtype=torch.long),
                    )
                )
                continue

            # Class cost (negative class probability for the target classes)
            out_prob = outputs["pred_logits"][b].softmax(-1)  # (Q, C+1)
            cost_class = -out_prob[:, tgt_labels]  # (Q, N_b)

            # Mask costs: downsample GT to the predicted-mask resolution
            tgt_masks_low = F.interpolate(
                tgt_masks_full.unsqueeze(1).float(),
                size=(mh, mw),
                mode="nearest",
            ).squeeze(1)  # (N_b, h, w)

            out_masks_flat = out_pred_masks[b].flatten(1).float()      # (Q, h*w)
            tgt_masks_flat = tgt_masks_low.flatten(1)                  # (N_b, h*w)

            cost_mask = _batch_sigmoid_ce_cost(out_masks_flat, tgt_masks_flat)
            cost_dice = _batch_dice_cost(out_masks_flat.sigmoid(), tgt_masks_flat)

            C = (
                self.cost_class * cost_class
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            )

            # scipy expects fp32 numpy on CPU
            row, col = linear_sum_assignment(C.detach().cpu().numpy())
            indices.append(
                (
                    torch.as_tensor(row, dtype=torch.long),
                    torch.as_tensor(col, dtype=torch.long),
                )
            )
        return indices
