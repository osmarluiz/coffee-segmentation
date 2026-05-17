"""
End-to-end smoke test for the Hungarian-matched Mask2Former pipeline.

Verifies:
  1. The model imports cleanly and instantiates with Combined-modality params
     (108 channels, 3 classes).
  2. forward returns a dict with the expected keys and shapes.
  3. The Hungarian matcher + SetCriterion run on real-shaped tensors.
  4. The full training step (autocast bf16 + backward) reduces the loss across
     a few iterations on a single batch (sanity check that the loss is
     actually optimisable, unlike the previous dense-CE recipe that pinned at
     log(3) forever).
"""

import contextlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.mask2former.matcher import HungarianMatcher
from segmentation_models_pytorch.decoders.mask2former.criterion import SetCriterion
from segmentation_models_pytorch.decoders.mask2former.training import build_m2f_targets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── 1) Construct ────────────────────────────────────────────────────────
    model = smp.Mask2Former(
        encoder_name="efficientnet-b7",
        encoder_weights=None,
        in_channels=108,
        classes=3,
        activation=None,                # logits at the head
        num_queries=100,
        num_decoder_layers=9,
        num_heads=8,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model built. Params: {n_params / 1e6:.2f}M")

    # ── 2) Forward returns a dict ──────────────────────────────────────────
    model.eval()
    x = torch.randn(2, 108, 512, 512, device=device)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, dict), f"expected dict, got {type(out)}"
    expected_keys = {"pred_logits", "pred_masks", "aux_outputs", "semantic"}
    assert set(out.keys()) == expected_keys, out.keys()
    assert out["pred_logits"].shape == (2, 100, 4), out["pred_logits"].shape
    assert out["pred_masks"].shape == (2, 100, 128, 128), out["pred_masks"].shape
    assert out["semantic"].shape == (2, 3, 512, 512), out["semantic"].shape
    assert len(out["aux_outputs"]) == 9, len(out["aux_outputs"])  # 9 layers
    for i, a in enumerate(out["aux_outputs"]):
        assert a["pred_logits"].shape == (2, 100, 4), (i, a["pred_logits"].shape)
        assert a["pred_masks"].shape == (2, 100, 128, 128), (i, a["pred_masks"].shape)
    print("forward dict OK:")
    print(f"  pred_logits: {tuple(out['pred_logits'].shape)}")
    print(f"  pred_masks:  {tuple(out['pred_masks'].shape)}")
    print(f"  aux_outputs: {len(out['aux_outputs'])} layers")
    print(f"  semantic:    {tuple(out['semantic'].shape)}")

    # ── 3) Criterion: build_targets + Hungarian + losses ───────────────────
    matcher = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=5.0)
    criterion = SetCriterion(
        num_classes=3,
        matcher=matcher,
        weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0},
        eos_coef=0.1,
        aux_loss=True,
        class_weight=[1.0, 15.0, 35.0],
    ).to(device)

    # Synthetic targets: STRUCTURED 3-quadrant mask (background top, coffee
    # bottom-left, eucalyptus bottom-right). Random labels are pathological for
    # the loss landscape: the model cannot fit noise, so we use a learnable
    # pattern instead.
    y = torch.zeros(2, 512, 512, dtype=torch.long, device=device)
    y[:, 256:, :256] = 1   # coffee bottom-left quadrant
    y[:, 256:, 256:] = 2   # eucalyptus bottom-right quadrant
    targets = build_m2f_targets(y)
    print(f"\ntargets:")
    for b, t in enumerate(targets):
        print(f"  image {b}: labels={t['labels'].tolist()}, "
              f"masks shape={tuple(t['masks'].shape)}")

    loss = criterion(out, targets)
    print(f"\ninitial loss (random model, random targets): {loss.item():.4f}")
    assert torch.isfinite(loss), "loss is not finite"

    # Show component losses
    print("  components:")
    for k, v in sorted(criterion.last_losses.items()):
        if "_aux_" in k:
            continue
        print(f"    {k}: {v.item():.4f}")

    # ── 4) Optimisable: 20 steps should reduce the loss on this batch ─
    print("\n--- training-step sanity check (20 iters on one batch) ---")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    losses = []
    for step in range(20):
        optimizer.zero_grad()
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with amp_ctx:
            out = model(x)
        loss = criterion(out, targets)
        loss.backward()
        # Gradient clipping helps with the first few large steps from random
        # init (the original M2F paper uses clip_grad_norm too).
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        optimizer.step()
        losses.append(loss.item())
        if step < 5 or step % 5 == 0:
            print(f"  step {step:2d}: loss={loss.item():.4f}")

    print(f"\ninitial -> final: {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"(delta {losses[-1] - losses[0]:+.4f})")
    # Pipeline is healthy if final-quarter mean is below initial-quarter mean
    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    print(f"first-5 avg: {early:.4f}; last-5 avg: {late:.4f}")
    assert late < early, "loss did NOT decrease — pipeline broken"

    print("\n[OK] Mask2Former Hungarian pipeline smoke test passed.")


if __name__ == "__main__":
    main()
