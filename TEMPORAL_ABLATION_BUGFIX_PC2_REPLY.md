# Re: TEMPORAL_ABLATION_BUGFIX.md — PC2 status

**TL;DR — PC2 was *not* hit by the `float32 is not JSON serializable` bug.**
We are running a different (older) standalone script. Pulled `59e2770` anyway
so future runs of the repo version will be safe.

---

## Why PC2 dodged the bug

PC1 ran the repo version: `cafe-segmentation/temporal_ablation.py` (commit
`6274bd6`), which delegates the training loop to
`smp.utils.train.TrainEpoch.run`. That returns `AverageValueMeter.mean`
values, which are `numpy.float32`. Those reached `save_status()` /
`json.dump()` and triggered the crash.

PC2 has been running an older standalone script that predates the repo
commit: `D:/projects/cafe/CODE/temporal_ablation.py`. It implements its own
training loop with hand-written accumulators:

```python
total_loss += loss.item()           # Python float (from .item())
return {'cross_entropy_loss': total_loss / n_batches,
        'miou':               total_miou  / n_batches}
```

`loss.item()` returns a Python `float`, so the values written to
`training_status.json` are native scalars. No serialization issue.

---

## Verification on PC2

```text
CSV rows:                                       13   (1 header + 12 data) ✓
"float32 is not JSON serializable" in log:      0    ✓
Truncated training_status.json files:           0    ✓
test_results.json files:                        11   (matches completed windows) ✓
```

The only `ERROR` ever logged on PC2 was the unrelated PyTorch 2.6 + CUDA 12.6
`Adam.foreach` bug (`'Tensor' object is not callable`) we already mitigate
with the watchdog. That one is a deterministic ~once-per-window-or-so crash,
not the JSON bug.

---

## What's in PC2's CSV right now (12 result rows)

| Window | Channels | Selection | mIoU | Coffee | Eucalyptus |
|--------|----------|-----------|------|--------|------------|
| 3-cons-Jan | 12 | val_loss | 73.4 | 81.8 | 42.9 |
| 3-cons-Jul | 12 | val_loss | 77.0 | 85.1 | 49.7 |
| 3-cons-Set | 12 | val_loss | 71.2 | 76.3 | 43.8 |
| 3-spread   | 12 | val_loss | 79.5 | 84.4 | 57.8 |
| 6-cons-1H  | 24 | val_loss | 78.8 | 85.9 | 53.7 |
| 6-cons-Apr | 24 | val_loss | 80.6 | 84.8 | 60.6 |
| 6-alt      | 24 | val_loss | 79.4 | 86.3 | 55.1 |
| 6-cons-2H  | 24 | val_loss | 75.7 | 85.4 | 45.5 |
| 9-cons     | 36 | val_loss | 74.6 | 83.1 | 44.6 |
| 9-cons     | 36 | val_miou | 81.9 | 87.7 | 60.8 |
| 9-cons-Dec | 36 | val_loss | 77.4 | 86.6 | 49.0 |
| 9-cons-Dec | 36 | val_miou | 84.1 | 88.1 | 67.1 |

Currently running: **3-cons-Apr** (epoch ~32/100). After it, **3-cons-Oct**
is queued. `12-full` was removed from PC2's queue because PC1 owns it. PC2
total ETA ≈ 18 hours from now.

---

## Things only PC2 has (extra, not requested by the reviewer)

- `3-cons-Set` (Sep–Nov, the "flowering" window) — early experimental.
- `3-spread` (Jan / Jul / Dec) — tests sparse temporal coverage; turned out
  to *beat* every consecutive 3-month window on Eucalyptus IoU (57.8% vs.
  ≤50%).
- `6-alt` (Jan/Mar/May/Jul/Sep/Nov) — every-other-month sampling; nearly
  ties the best 6-month consecutive window despite covering the whole year.

These three are not in the repo-version `WINDOWS` list — they are bonus
analyses, useful for the discussion section but not for the reviewer's
direct ask. PC1 reproducing them is optional.

---

## Things only PC1 has (so far)

- `12-full` with bfloat16 AMP — the internal-baseline run that lets the
  table be 100% internally consistent (vs. quoting 86.10% from the paper
  which was trained without AMP and without dual save).
- The bugfix that everyone else (anyone using the repo script) will need.

---

## Pulled `59e2770` on PC2

Local checkout is now up to date. If PC2 ever switches from its standalone
`CODE/temporal_ablation.py` to the repo `cafe-segmentation/temporal_ablation.py`,
the fix is already there — no second migration needed.

Thanks for the catch and the post-mortem.
