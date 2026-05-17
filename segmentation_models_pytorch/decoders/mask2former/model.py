"""
SMP-compatible Mask2Former wrapper.

The decoder produces per-query class logits (B, Q, C+1) and mask logits
(B, Q, H/4, W/4) plus a list of intermediate predictions for deep supervision.
This wrapper:

  * Always returns a dict from ``forward`` so the Hungarian-matched
    SetCriterion can compute the loss on the rich per-query predictions.
  * Includes a ``'semantic'`` field in that dict containing the
    (B, C, H, W) dense semantic map (softmax(class) without the no-object
    channel, contracted with sigmoid(mask), then upsampled 4x). Downstream
    code (mIoU metric, evaluate.py, predict.py) reads this field.

This breaks the standard SMP ``model(x) -> tensor`` contract on purpose:
Mask2Former is a set-prediction model and the Hungarian loss needs the full
per-query view, not just the rasterised semantic output.
"""

from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import SegmentationModel, ClassificationHead
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading
from segmentation_models_pytorch.base.utils import is_torch_compiling
from segmentation_models_pytorch.encoders import get_encoder

from .decoder import Mask2FormerDecoder


class Mask2Former(SegmentationModel):
    """Mask2Former (Cheng et al., 2022) integrated into the SMP API."""

    requires_divisible_input_shape = True

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_dim: int = 256,
        num_queries: int = 100,
        num_decoder_layers: int = 9,
        num_heads: int = 8,
        pixel_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.num_classes = classes

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        self.decoder = Mask2FormerDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            num_classes=classes,
            decoder_dim=decoder_dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            pixel_decoder_layers=pixel_decoder_layers,
            dim_feedforward=dim_feedforward,
        )

        # The segmentation head only restores full resolution (4x bilinear) and
        # applies the final activation. The class projection lives in the
        # decoder's class_head; no conv here.
        self.segmentation_head = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=4),
            Activation(activation),
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "mask2former-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    @staticmethod
    def _semantic_from(pred_logits: torch.Tensor, pred_masks: torch.Tensor) -> torch.Tensor:
        """Paper Eq. 5: dense semantic = einsum(softmax(class)[:-1], sigmoid(mask))."""
        class_prob = F.softmax(pred_logits, dim=-1)[..., :-1]  # drop no-object
        mask_prob = pred_masks.sigmoid()
        return torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)

    def forward(self, x: torch.Tensor) -> dict:
        """Always returns a dict.

        Keys:
          - ``pred_logits``: (B, Q, C+1) class logits, FINAL decoder layer.
          - ``pred_masks``:  (B, Q, H/4, W/4) mask logits, FINAL decoder layer.
          - ``aux_outputs``: list of dicts (each with the same two keys) from
            intermediate decoder layers, for deep supervision.
          - ``semantic``: (B, C, H, W) dense semantic prediction at full
            resolution, suitable for argmax-based metrics and inference.
        """
        if not (
            torch.jit.is_scripting() or torch.jit.is_tracing() or is_torch_compiling()
        ):
            self.check_input_shape(x)

        features = self.encoder(x)
        dec_out = self.decoder(features)

        semantic_low = self._semantic_from(
            dec_out["pred_logits"], dec_out["pred_masks"]
        )
        semantic_full = self.segmentation_head(semantic_low)

        return {
            "pred_logits": dec_out["pred_logits"],
            "pred_masks": dec_out["pred_masks"],
            "aux_outputs": dec_out["aux_outputs"],
            "semantic": semantic_full,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: return only the dense semantic tensor (B, C, H, W)."""
        if self.training:
            self.eval()
        return self.forward(x)["semantic"]

    def eval(self):
        """Switch to eval mode but keep BatchNorm layers in train mode.

        Why: when training Mask2Former from scratch with a small batch
        (batch_size=8) and an SMP encoder full of BatchNorm layers like
        efficientnet-b7, the BN running mean/var take many epochs to
        stabilise. During the first dozens of epochs the eval-mode BN values
        diverge wildly from the train-mode batch statistics, which means the
        forward pass at validation time produces almost-random features even
        though the model is learning fine in train mode (e.g. epoch 1 here:
        train mIoU 0.38, val mIoU 0.002, val loss 7x higher than train).

        Forcing BN to keep using the current batch's stats during val and
        test fixes this measurement gap. The same applies to inference at
        deployment (predict.py) until the running stats converge.
        """
        super().eval()
        for m in self.modules():
            if isinstance(
                m,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.SyncBatchNorm,
                ),
            ):
                m.train()
        return self
