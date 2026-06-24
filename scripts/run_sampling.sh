#!/bin/bash
# =============================================================================
# run_sampling.sh — Sample images from a MoE checkpoint
#
# Usage:
#   bash scripts/run_sampling.sh                              # sample from best checkpoint
#   bash scripts/run_sampling.sh --ckpt path/to/ckpt.pt      # specific checkpoint
#   bash scripts/run_sampling.sh --steps 250 --seed 123      # custom steps/seed
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_sampling.sh      # use GPU 1
# =============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
BEST_CKPT=$(ls -t results-finetune/004-DiT-XL-2-MoE-finetune-moe24-27/checkpoints/*.pt 2>/dev/null | head -1)
CKPT="${BEST_CKPT:-pretrained_models/DiT-XL-2-256x256.pt}"
MOE_BLOCKS="24 25 26 27"
NUM_EXPERTS=4
STEPS=250
SEED=42
OUTPUT="sample_output_$(date +%Y%m%d_%H%M%S).png"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt) CKPT="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --num-experts) NUM_EXPERTS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "=== CD-MoE Sampling ==="
echo "  Checkpoint:  $CKPT"
echo "  MoE blocks:  $MOE_BLOCKS"
echo "  Num experts: $NUM_EXPERTS"
echo "  Steps:       $STEPS"
echo "  Seed:        $SEED"
echo "  Output:      $OUTPUT"
echo ""

conda run -n dit python sample_diffmoe.py \
    --ckpt "$CKPT" \
    --moe-blocks $MOE_BLOCKS \
    --num-experts $NUM_EXPERTS \
    --num-sampling-steps $STEPS \
    --seed $SEED \
    --output "$OUTPUT"

echo ""
echo "[✓] Saved: $OUTPUT"
