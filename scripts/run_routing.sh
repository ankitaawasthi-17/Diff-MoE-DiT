#!/bin/bash
# =============================================================================
# run_routing.sh — Run routing analysis on a checkpoint
#
# Usage:
#   bash scripts/run_routing.sh                          # analyze best checkpoint
#   bash scripts/run_routing.sh --ckpt path/to/ckpt.pt  # specific checkpoint
#   bash scripts/run_routing.sh --pretrained             # analyze pretrained (no finetune)
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_routing.sh  # use GPU 1
# =============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BEST_CKPT=$(ls -t results-finetune/004-DiT-XL-2-MoE-finetune-moe24-27/checkpoints/*.pt 2>/dev/null | head -1)
CKPT="${BEST_CKPT}"
MOE_BLOCKS="24 25 26 27"
NUM_EXPERTS=4
STEPS=50
SEED=42
OUTPUT_DIR="routing_analysis/run_$(date +%Y%m%d_%H%M%S)"
PRETRAINED=false

# Mixed categories (default) — animals, nature, objects
CLASS_LABELS="207 360 387 974 88 979 417 279"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)         CKPT="$2"; shift 2 ;;
        --pretrained)   CKPT=""; PRETRAINED=true; shift ;;
        --num-experts)  NUM_EXPERTS="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --steps)        STEPS="$2"; shift 2 ;;
        --class-labels) CLASS_LABELS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "=== CD-MoE Routing Analysis ==="
if $PRETRAINED; then
    echo "  Mode:        PRETRAINED (no finetuning)"
else
    echo "  Checkpoint:  $CKPT"
fi
echo "  MoE blocks:  $MOE_BLOCKS"
echo "  Num experts: $NUM_EXPERTS"
echo "  Steps:       $STEPS"
echo "  Classes:     $CLASS_LABELS"
echo "  Output:      $OUTPUT_DIR"
echo ""

CKPT_ARG=""
if [ -n "$CKPT" ]; then
    CKPT_ARG="--ckpt $CKPT"
fi

conda run -n dit python analyze_routing.py \
    $CKPT_ARG \
    --moe-blocks $MOE_BLOCKS \
    --num-experts $NUM_EXPERTS \
    --num-sampling-steps $STEPS \
    --seed $SEED \
    --class-labels $CLASS_LABELS \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "[✓] Results saved to: $OUTPUT_DIR"
