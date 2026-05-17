"""
Mask2Former decoder (Cheng et al., CVPR 2022).

This is a faithful re-implementation of the Mask2Former architecture for
semantic segmentation, written in pure PyTorch (no MSDeformAttn CUDA extension).

Differences from the original paper:
  - Pixel decoder uses standard transformer encoder layers on strides 16+32
    only, not Multi-Scale Deformable Attention. Stride 8 and stride 4 features
    are produced by FPN top-down without self-attention enrichment. This keeps
    memory bounded on a single GPU and avoids the C++/CUDA build.
  - Training loss is dense weighted CrossEntropy on the combined semantic
    output (class x mask einsum), NOT Hungarian matching with mask BCE/dice
    per query. This is intentional, to give a fair comparison with the other
    SMP architectures in this repo that all share the same loss.
"""

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple multi-layer perceptron used for the mask embedding head."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x, inplace=True)
        return x


class SinusoidalPositionalEncoding2D(nn.Module):
    """Standard 2D sinusoidal positional encoding (DETR-style)."""

    def __init__(self, num_pos_feats: int = 128, temperature: float = 10000.0):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — only shape used; values ignored
        B, _, H, W = x.shape
        device = x.device
        not_mask = torch.ones((B, H, W), device=device, dtype=torch.float32)
        y_embed = not_mask.cumsum(dim=1)
        x_embed = not_mask.cumsum(dim=2)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed.unsqueeze(-1) / dim_t  # (B, H, W, num_pos_feats)
        pos_y = y_embed.unsqueeze(-1) / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
        ).flatten(-2)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
        ).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1).permute(0, 3, 1, 2)  # (B, 2*npf, H, W)
        return pos


class PixelDecoder(nn.Module):
    """Pixel decoder: produces multi-scale features and a high-resolution mask feature.

    Inputs are the last 4 encoder feature maps (strides 4, 8, 16, 32).
    Outputs:
      - mask_features at stride 4, used to predict mask logits via einsum.
      - multi-scale features at strides 32, 16, 8 (ordered low->high res) for
        the transformer decoder cross-attention.
      - positional encodings for each multi-scale feature.

    Transformer encoder is applied to strides 32 and 16 only (concatenated
    tokens, jointly self-attended). Strides 8 and 4 are produced by FPN
    top-down with 3x3 convs.
    """

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_dim: int = 256,
        num_transformer_layers: int = 3,
        num_heads: int = 8,
    ):
        super().__init__()
        # Use last 4 encoder feature maps (strides 4, 8, 16, 32)
        if len(encoder_channels) < 4:
            raise ValueError(
                f"Mask2Former pixel decoder needs at least 4 encoder feature scales, "
                f"got {len(encoder_channels)}."
            )
        c4, c8, c16, c32 = encoder_channels[-4:]

        # Lateral 1x1 projections to common decoder dim
        self.lateral_4 = nn.Conv2d(c4, decoder_dim, kernel_size=1)
        self.lateral_8 = nn.Conv2d(c8, decoder_dim, kernel_size=1)
        self.lateral_16 = nn.Conv2d(c16, decoder_dim, kernel_size=1)
        self.lateral_32 = nn.Conv2d(c32, decoder_dim, kernel_size=1)

        # GroupNorm for the projected features
        self.gn_4 = nn.GroupNorm(32, decoder_dim)
        self.gn_8 = nn.GroupNorm(32, decoder_dim)
        self.gn_16 = nn.GroupNorm(32, decoder_dim)
        self.gn_32 = nn.GroupNorm(32, decoder_dim)

        # Self-attention transformer encoder on the lowest 2 scales (32 + 16)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # FPN top-down convs for strides 8 and 4 (after fusing with upsampled features)
        self.fpn_8 = nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1)
        self.fpn_4 = nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1)

        # Level embeddings for the 2 attended scales (added to features as token-type markers)
        self.level_embed = nn.Embedding(2, decoder_dim)

        # 2D positional encoding (output dim = 2 * num_pos_feats = decoder_dim)
        self.pos_enc = SinusoidalPositionalEncoding2D(num_pos_feats=decoder_dim // 2)

    def forward(self, features: List[torch.Tensor]) -> Dict:
        # features comes from the encoder; we use the last 4 maps (strides 4..32)
        f4_raw, f8_raw, f16_raw, f32_raw = features[-4:]

        f4 = F.relu(self.gn_4(self.lateral_4(f4_raw)), inplace=True)
        f8 = F.relu(self.gn_8(self.lateral_8(f8_raw)), inplace=True)
        f16 = F.relu(self.gn_16(self.lateral_16(f16_raw)), inplace=True)
        f32 = F.relu(self.gn_32(self.lateral_32(f32_raw)), inplace=True)

        # Positional encodings (same dim as features)
        pos_16 = self.pos_enc(f16)
        pos_32 = self.pos_enc(f32)

        # Flatten + add level embeddings for the 2 attended scales
        B = f32.shape[0]
        lvl_32 = self.level_embed.weight[0].view(1, -1, 1, 1)
        lvl_16 = self.level_embed.weight[1].view(1, -1, 1, 1)

        h32, w32 = f32.shape[-2:]
        h16, w16 = f16.shape[-2:]
        n32, n16 = h32 * w32, h16 * w16

        tokens_32 = (f32 + lvl_32 + pos_32).flatten(2).transpose(1, 2)  # (B, n32, D)
        tokens_16 = (f16 + lvl_16 + pos_16).flatten(2).transpose(1, 2)  # (B, n16, D)
        tokens = torch.cat([tokens_32, tokens_16], dim=1)  # (B, n32+n16, D)

        tokens = self.transformer(tokens)

        # Split back and reshape to 2D feature maps
        t32, t16 = tokens.split([n32, n16], dim=1)
        f32_out = t32.transpose(1, 2).reshape(B, -1, h32, w32)
        f16_out = t16.transpose(1, 2).reshape(B, -1, h16, w16)

        # FPN top-down: enrich f8 with f16_out, then f4 with f8_out
        f8_out = self.fpn_8(
            f8 + F.interpolate(f16_out, size=f8.shape[-2:], mode="bilinear", align_corners=False)
        )
        f4_out = self.fpn_4(
            f4 + F.interpolate(f8_out, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        )

        return {
            "mask_features": f4_out,
            "multi_scale": [f32_out, f16_out, f8_out],
            "spatial_shapes": [(h32, w32), (h16, w16), f8_out.shape[-2:]],
        }


class TransformerDecoderLayer(nn.Module):
    """One Mask2Former transformer decoder layer.

    Order per the paper (Sec. 3.1): masked cross-attn -> self-attn -> FFN.
    All sub-layers use post-norm.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,        # (B, Q, D)
        query_pos: torch.Tensor,      # (B, Q, D)
        memory: torch.Tensor,         # (B, S, D) — flattened features at one scale
        memory_pos: torch.Tensor,     # (B, S, D)
        attn_mask: torch.Tensor = None,  # (B*nheads, Q, S) bool — True means BLOCK
    ) -> torch.Tensor:
        # 1) Masked cross-attention: queries attend to features
        q = queries + query_pos
        k = memory + memory_pos
        v = memory
        attn_out, _ = self.cross_attn(q, k, v, attn_mask=attn_mask, need_weights=False)
        queries = self.norm1(queries + self.dropout1(attn_out))

        # 2) Self-attention among queries
        q = queries + query_pos
        attn_out, _ = self.self_attn(q, q, queries, need_weights=False)
        queries = self.norm2(queries + self.dropout2(attn_out))

        # 3) FFN
        ff = self.linear2(self.dropout3(F.relu(self.linear1(queries), inplace=True)))
        queries = self.norm3(queries + ff)

        return queries


class Mask2FormerDecoder(nn.Module):
    """Mask2Former decoder.

    Produces dense semantic logits at stride 4: shape (B, num_classes, H/4, W/4).
    The SegmentationModel wrapper upsamples them by 4x to recover full resolution.

    Args:
        encoder_channels: channel counts of the encoder's feature pyramid
            (one entry per scale, including input). The last 4 entries are used.
        encoder_depth: kept for API compatibility with SMP (must be >= 4).
        num_classes: number of segmentation classes (the model's output dim).
        decoder_dim: transformer hidden size (default 256).
        num_queries: number of object queries (default 100, paper default).
        num_decoder_layers: total transformer decoder layers (default 9 = 3 rounds x 3 scales).
        num_heads: attention heads per layer (default 8).
        pixel_decoder_layers: transformer encoder layers in the pixel decoder (default 3).
    """

    def __init__(
        self,
        encoder_channels: List[int],
        encoder_depth: int = 5,
        num_classes: int = 3,
        decoder_dim: int = 256,
        num_queries: int = 100,
        num_decoder_layers: int = 9,
        num_heads: int = 8,
        pixel_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        if encoder_depth < 4:
            raise ValueError(
                f"Mask2Former requires encoder_depth >= 4 to have 4 feature scales, "
                f"got {encoder_depth}."
            )
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.decoder_dim = decoder_dim

        self.pixel_decoder = PixelDecoder(
            encoder_channels=encoder_channels,
            decoder_dim=decoder_dim,
            num_transformer_layers=pixel_decoder_layers,
            num_heads=num_heads,
        )

        # Learnable query content + learnable query positional embeddings
        self.query_feat = nn.Embedding(num_queries, decoder_dim)
        self.query_pos = nn.Embedding(num_queries, decoder_dim)

        # Level embedding for the 3 multi-scale features in the transformer decoder
        self.dec_level_embed = nn.Embedding(3, decoder_dim)

        self.decoder_layers = nn.ModuleList(
            TransformerDecoderLayer(
                d_model=decoder_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        )

        self.decoder_norm = nn.LayerNorm(decoder_dim)
        # Class head outputs C + 1 logits: the extra one is the "no-object"
        # class used by the Hungarian matching loss for unmatched queries.
        self.class_head = nn.Linear(decoder_dim, num_classes + 1)
        self.mask_embed_head = MLP(decoder_dim, decoder_dim, decoder_dim, num_layers=3)

        self.pos_enc = SinusoidalPositionalEncoding2D(num_pos_feats=decoder_dim // 2)

    def _heads(self, queries: torch.Tensor, mask_features: torch.Tensor):
        """Compute class logits and mask logits from queries."""
        q = self.decoder_norm(queries)
        class_logits = self.class_head(q)                      # (B, Q, C)
        mask_embed = self.mask_embed_head(q)                   # (B, Q, D)
        mask_logits = torch.einsum("bqd,bdhw->bqhw", mask_embed, mask_features)
        return class_logits, mask_logits

    def _attn_mask(self, mask_logits: torch.Tensor, target_size) -> torch.Tensor:
        """Build the cross-attention mask for the next decoder layer.

        Returns shape (B*num_heads, Q, S) bool, True means BLOCK.
        Rows that would block ALL keys are forced to False to avoid softmax NaN.
        """
        H_t, W_t = target_size
        with torch.no_grad():
            m = F.interpolate(
                mask_logits, size=(H_t, W_t), mode="bilinear", align_corners=False
            )
            # cast to float32 for stable sigmoid threshold even under autocast
            m = m.float().sigmoid()
        attn_mask = (m < 0.5).flatten(2)  # (B, Q, S) bool

        B, Q, S = attn_mask.shape
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn_mask = attn_mask.reshape(B * self.num_heads, Q, S)

        # Force False where the whole row would be blocked (all keys masked out)
        all_blocked = attn_mask.all(dim=-1, keepdim=True)
        attn_mask = torch.where(all_blocked, torch.zeros_like(attn_mask), attn_mask)
        return attn_mask

    def forward(self, features: List[torch.Tensor]) -> dict:
        """Returns a dict of per-layer predictions for Hungarian-matched losses.

        Keys:
          - ``pred_logits``: (B, Q, C+1) class logits from the FINAL decoder
            layer.
          - ``pred_masks``:  (B, Q, H/4, W/4) mask logits from the FINAL layer.
          - ``aux_outputs``: list of dicts (same two keys) from the initial
            (pre-layer-0) prediction plus each intermediate decoder layer,
            for deep supervision.

        The semantic post-processing (sum over queries of softmax(class) x
        sigmoid(mask)) is performed in the SegmentationModel wrapper.
        """
        pd_out = self.pixel_decoder(features)
        mask_features: torch.Tensor = pd_out["mask_features"]  # (B, D, H/4, W/4)
        multi_scale: List[torch.Tensor] = pd_out["multi_scale"]
        spatial_shapes = pd_out["spatial_shapes"]

        B = mask_features.shape[0]

        # Pre-flatten multi-scale features and positional encodings
        flat_mem, flat_pos = [], []
        for i, f in enumerate(multi_scale):
            pos = self.pos_enc(f)
            lvl = self.dec_level_embed.weight[i].view(1, -1, 1, 1)
            flat_mem.append((f + lvl).flatten(2).transpose(1, 2))   # (B, S_i, D)
            flat_pos.append(pos.flatten(2).transpose(1, 2))         # (B, S_i, D)

        # Initial queries
        queries = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1).contiguous()
        query_pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Initial prediction (used to build the attn mask for layer 0)
        class_logits, mask_logits = self._heads(queries, mask_features)
        predictions = [{"pred_logits": class_logits, "pred_masks": mask_logits}]

        for layer_idx in range(self.num_decoder_layers):
            scale_idx = layer_idx % 3
            attn_mask = self._attn_mask(mask_logits, spatial_shapes[scale_idx])
            queries = self.decoder_layers[layer_idx](
                queries=queries,
                query_pos=query_pos,
                memory=flat_mem[scale_idx],
                memory_pos=flat_pos[scale_idx],
                attn_mask=attn_mask,
            )
            class_logits, mask_logits = self._heads(queries, mask_features)
            predictions.append({"pred_logits": class_logits, "pred_masks": mask_logits})

        return {
            "pred_logits": predictions[-1]["pred_logits"],
            "pred_masks": predictions[-1]["pred_masks"],
            "aux_outputs": predictions[:-1],
        }
