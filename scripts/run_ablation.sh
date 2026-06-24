#!/bin/bash
# =============================================================================
# run_ablation.sh — E1–E4 ablation: 1/2/3/4 experts × pretrained/finetuned
#                   × mixed/single category prompts
#
# From meeting notes: compare routing behavior across expert counts,
# both before and after finetuning, to answer:
#   Q1: How much does routing change before vs after finetuning?
#   Q2: Does finetuning produce concept specialization?
#
# Usage:
#   bash scripts/run_ablation.sh                   # run full ablation matrix
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_ablation.sh   # use GPU 1
#
# Output structure:
#   ablation_results/
#     pretrained_1exp_mixed/
#     pretrained_1exp_single/
#     pretrained_2exp_mixed/
#     pretrained_2exp_single/
#     pretrained_3exp_mixed/
#     pretrained_3exp_single/
#     pretrained_4exp_mixed/
#     pretrained_4exp_single/
#     finetuned_4exp_mixed/
#     finetuned_4exp_single/
# =============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Best finetuned checkpoint (4 experts)
FINETUNED_CKPT=$(ls -t results-finetune/004-DiT-XL-2-MoE-finetune-moe24-27/checkpoints/*.pt 2>/dev/null | head -1)
PRETRAINED_CKPT="pretrained_models/DiT-XL-2-256x256.pt"
MOE_BLOCKS="24 25 26 27"
STEPS=50
SEED=42
OUTPUT_BASE="ablation_results"
mkdir -p "$OUTPUT_BASE"

# ── ImageNet class label sets ─────────────────────────────────────────────────
# Mixed categories (animals + nature + objects — as in meeting notes)
# 207=golden retriever, 360=otter, 387=red panda, 974=geyser
# 88=macaw/parrot, 979=balloon, 417=balloon2, 279=arctic fox
MIXED_LABELS="207 360 387 974 88 979 417 279"

# Single category — animals only
# 207=golden retriever, 360=otter, 387=red panda, 281=tabby cat
# 388=leopard, 340=zebra, 330=koala, 386=giant panda
SINGLE_LABELS="207 360 387 281 388 340 330 386"

echo "╔══════════════════════════════════════════════════════╗"
echo "║     CD-MoE Expert Ablation Study (E1–E4)            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Finetuned checkpoint: $FINETUNED_CKPT"
echo "Output dir:           $OUTPUT_BASE"
echo ""

# ── Helper function ───────────────────────────────────────────────────────────
run_analysis() {
    local LABEL="$1"       # e.g. "pretrained_2exp_mixed"
    local CKPT_ARG="$2"   # e.g. "--ckpt path/to/ckpt.pt" or "" for pretrained
    local N_EXP="$3"      # number of experts
    local CLASSES="$4"    # space-separated class label string
    local OUT_DIR="$OUTPUT_BASE/$LABEL"

    echo "──────────────────────────────────────────────────────"
    echo "  Running: $LABEL"
    echo "  Experts: $N_EXP | Classes: $CLASSES"
    echo "  Output:  $OUT_DIR"

    mkdir -p "$OUT_DIR"

    conda run -n dit python analyze_routing.py \
        $CKPT_ARG \
        --moe-blocks $MOE_BLOCKS \
        --num-experts "$N_EXP" \
        --num-sampling-steps "$STEPS" \
        --seed "$SEED" \
        --class-labels $CLASSES \
        --output-dir "$OUT_DIR" \
        2>&1 | tee "$OUT_DIR/run.log"

    if [ $? -eq 0 ]; then
        echo "  [✓] Done: $LABEL"
    else
        echo "  [✗] FAILED: $LABEL — check $OUT_DIR/run.log"
    fi
    echo ""
}

# ── E1–E4: PRETRAINED model (random gate init) ────────────────────────────────
# Note: for pretrained runs, num_experts can be varied freely since
# no trained checkpoint is needed — gate weights are randomly initialized.
# This shows baseline routing behavior without finetuning.

echo "=== PRETRAINED MODEL RUNS (random MoE gate) ==="
echo ""

for N_EXP in 1 2 3 4; do
    # Mixed categories
    run_analysis "pretrained_${N_EXP}exp_mixed" "" "$N_EXP" "$MIXED_LABELS"
    # Single category (all animals)
    run_analysis "pretrained_${N_EXP}exp_single" "" "$N_EXP" "$SINGLE_LABELS"
done

# ── E4 FINETUNED model (105k checkpoint, 4 experts) ──────────────────────────
# Note: finetuned checkpoint only exists for 4 experts.
# Running 1/2/3 experts with finetuned checkpoint would be invalid —
# the gate weights were trained with 4-expert architecture.

echo "=== FINETUNED MODEL RUNS (4 experts, 105k checkpoint) ==="
echo ""

if [ -z "$FINETUNED_CKPT" ]; then
    echo "[!] No finetuned checkpoint found — skipping finetuned runs"
    echo "    Expected: results-finetune/004-.../checkpoints/*.pt"
else
    run_analysis "finetuned_4exp_mixed" "--ckpt $FINETUNED_CKPT" "4" "$MIXED_LABELS"
    run_analysis "finetuned_4exp_single" "--ckpt $FINETUNED_CKPT" "4" "$SINGLE_LABELS"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════╗"
echo "║                   Ablation Complete                  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: $OUTPUT_BASE/"
ls "$OUTPUT_BASE/" | sed 's/^/  /'
echo ""
echo "Next step: generate comparison plots"
echo "  conda run -n dit python ablation_compare.py --results-dir $OUTPUT_BASE"
