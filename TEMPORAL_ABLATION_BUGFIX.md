# Temporal Ablation — JSON serialization bug (FIXED in commit `59e2770`)

**For PC2 / anyone running `temporal_ablation.py`: please verify whether your
machine hit this bug, then `git pull`.**

---

## The bug

`temporal_ablation.py` (as originally committed in `6274bd6`) crashes at the
**end of epoch 1 of every window** with:

```
TypeError: Object of type float32 is not JSON serializable
```

Cause: `save_status()` and the `test_results.json` dump passed
`numpy.float32` values (the validation loss / mIoU come from
`AverageValueMeter.mean` via `valid_logs`) straight into `json.dump()`.
`json.dump` cannot serialize numpy scalar types. This is deterministic and
environment-independent — it happens on every machine, every window
(3 / 6 / 9 / 12-month), because the failure is in `save_status`, which is
window-size-agnostic.

## Why it is easy to miss

`main()` wraps each window in a `try/except`. When a window crashes, the
exception is **caught and logged as `ERROR`**, the loop moves to the next
window, and at the end the script still prints:

```
TEMPORAL ABLATION COMPLETE
```

and exits with **code 0**. So a fully-failed run looks superficially like a
successful one.

## How to check whether your machine was affected

Run these on the machine that executed `temporal_ablation.py`:

1. **Inspect the results CSV:**
   ```
   models/temporal_ablation/temporal_ablation_results.csv
   ```
   - If it is **missing or has no data rows** → every window failed.
   - If it has one/few rows but not all expected windows → partial failure.

2. **Grep the ablation log:**
   ```bash
   grep -i "float32 is not JSON serializable" models/temporal_ablation/ablation.log
   grep -i "ERROR on window" models/temporal_ablation/ablation.log
   ```
   Any hit confirms the bug was triggered.

3. **Check per-window output dirs** under
   `models/temporal_ablation/Planet_Segformer_efficientnet-b7_<window>/`:
   - A corrupted `training_status.json` truncated mid-line, e.g.
     `{"last_completed_epoch": 0, "best_loss": ` (no closing brace), is the
     signature of the crash.
   - `best_model_loss.pth` / `best_model_miou.pth` / `last_model.pth` may
     still exist (they are saved *before* `save_status`), but
     `test_results.json` will be missing.

## The fix

Commit `59e2770` fixes `temporal_ablation.py`:

- `save_status()` — casts epoch / loss / mIoU fields to native `int` / `float`.
- `test_results.json` dump — casts `best_val_loss` / `best_val_miou` to `float`.
- `load_status()` — tolerates a truncated/corrupted `training_status.json`
  (returns a clean default instead of raising `JSONDecodeError`), so a
  re-run resumes the window cleanly.

No training logic was changed — only the checkpoint/status I/O.

## What PC2 should do

```bash
git pull origin main          # gets commit 59e2770
```

Then re-run as planned:

```bash
python temporal_ablation.py --all --amp bfloat16
```

The fixed `load_status` makes this safe even if a previous (buggy) run left
corrupted `training_status.json` files behind — those windows simply restart
from epoch 0. If you want a guaranteed-clean slate, delete the affected
`models/temporal_ablation/Planet_Segformer_efficientnet-b7_<window>/`
directories before re-running; windows already present (and complete) in
`temporal_ablation_results.csv` are skipped automatically.

## Status of the PC1 run

PC1's `12-full` window hit this bug on its first launch, was fixed, and is
now running on the corrected code. No action needed for `12-full`.
