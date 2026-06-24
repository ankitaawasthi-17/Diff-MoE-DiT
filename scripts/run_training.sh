#!/bin/bash
# =============================================================================
# run_training.sh — Launch MoE finetuning with auto-restart watchdog
#
# Usage:
#   bash scripts/run_training.sh                        # default 4 experts, blocks 24-27
#   bash scripts/run_training.sh --num-experts 8        # override experts
#   bash scripts/run_training.sh --moe-blocks 20 21 22 23 24 25 26 27
# =============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults (override via CLI args) ─────────────────────────────────────────
DATA_PATH="/local/a/imagenet/imagenet2012/train/"
CKPT="pretrained_models/DiT-XL-2-256x256.pt"
MOE_BLOCKS="24 25 26 27"
NUM_EXPERTS=4
EPOCHS=1
BATCH_SIZE=16
LR=1e-4
LOG_EVERY=100
CKPT_EVERY=5000

# Pass-through any extra args directly to finetune_moe.py
EXTRA_ARGS="$@"

echo "=== CD-MoE Training ==="
echo "  Data:        $DATA_PATH"
echo "  Checkpoint:  $CKPT"
echo "  MoE blocks:  $MOE_BLOCKS"
echo "  Num experts: $NUM_EXPERTS"
echo "  Epochs:      $EPOCHS"
echo "  Extra args:  $EXTRA_ARGS"
echo ""

# ── Check conda env ───────────────────────────────────────────────────────────
if ! conda run -n dit python -c "import torch" 2>/dev/null; then
    echo "ERROR: conda env 'dit' not found. Run: bash scripts/setup_env.sh"
    exit 1
fi

# ── Launch via watchdog (handles crashes + auto-resume) ──────────────────────
echo "[+] Starting training watchdog (handles auto-resume on crash)..."
echo "    Log: watchdog.log"
echo "    Training logs: training_auto_resume_*.log"
echo ""

nohup bash training_watchdog.sh > watchdog.log 2>&1 &
WATCHDOG_PID=$!
disown -h

echo "[✓] Watchdog started (PID: $WATCHDOG_PID)"
echo ""
echo "Monitor with:"
echo "  tail -f watchdog.log"
echo "  tail -f \$(ls -t training_auto_resume_*.log | head -1)"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
