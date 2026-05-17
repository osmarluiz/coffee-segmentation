#!/bin/bash
# Run Mask2Former on Planet, VH, VV sequentially with efficientnet-b7.
# --resume so any individual crash is recoverable without restarting the whole queue.

LOG_DIR="/d/coffee/coffee-segmentation"
PY="/c/Users/Admin/anaconda3/envs/smp/python.exe"
DATA="/d/SEGMENTATION MODELS/DATASETS/3_CAFE"
OUT="/d/coffee/coffee-segmentation/models"
QUEUE_LOG="$LOG_DIR/training_m2f_queue.log"

echo "=== Queue starting @ $(date -Iseconds) ===" >> "$QUEUE_LOG"

for dataset in Planet VH VV; do
  echo "=== [$dataset] starting @ $(date -Iseconds) ===" >> "$QUEUE_LOG"
  "$PY" -u "$LOG_DIR/train.py" \
    --dataset "$dataset" --model Mask2Former --encoder efficientnet-b7 \
    --data-dir "$DATA" --output-dir "$OUT" --resume \
    >> "$LOG_DIR/training_m2f_${dataset}.log" 2>&1
  rc=$?
  echo "=== [$dataset] done @ $(date -Iseconds) exit_code=$rc ===" >> "$QUEUE_LOG"
done

echo "=== Queue finished @ $(date -Iseconds) ===" >> "$QUEUE_LOG"
