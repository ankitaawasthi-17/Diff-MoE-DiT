#!/bin/bash
CHECKPOINT_DIR="results-finetune/003-DiT-XL-2-MoE-finetune-moe24-27/checkpoints"
KEEP=2
INTERVAL_SECONDS=300
while true; do
    echo ""
    echo "=== Check at $(date) ==="
    python cleanup_checkpoints.py --dir "$CHECKPOINT_DIR" --keep "$KEEP" --confirm
    echo "--- Current quota status ---"
    df -h ~ | tail -1
    sleep "$INTERVAL_SECONDS"
done
