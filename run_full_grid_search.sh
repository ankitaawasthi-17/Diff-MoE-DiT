#!/bin/bash
# run_full_grid_search.sh — Runs all 3 phases + full training automatically.
# Usage: bash run_full_grid_search.sh

set -e

# Setup
source /home/min/a/awasthi9/miniconda3/etc/profile.d/conda.sh
conda activate dit
cd /home/nano01/a/awasthi9/Research/Diffusion_MoE/CD_MOE/DiT

BASE_CKPT="results-finetune/004-DiT-XL-2-MoE-finetune-moe24-27/checkpoints/0115000.pt"
DATA_PATH="/local/a/imagenet/imagenet2012/train/"
STEPS=10000

echo "============================================================"
echo "  CD-MoE FULL GRID SEARCH"
echo "  Started: $(date)"
echo "============================================================"

# ── Phase 1: Diversity Weight Sweep ────────────────────────────
echo ""
echo ">>> PHASE 1: diversity_weight sweep"
python grid_search_cdmoe.py \
    --base-ckpt $BASE_CKPT \
    --data-path $DATA_PATH \
    --phase 1 \
    --train-steps $STEPS

# Extract best diversity_weight from Phase 1 results
BEST_DW=$(python -c "
import json
with open('grid_search_results/phase1_summary.json') as f:
    results = json.load(f)
best = max(results, key=lambda x: x['summary']['composite_score'])
print(best['config']['diversity_weight'])
")
echo ""
echo ">>> Phase 1 BEST diversity_weight = $BEST_DW"

# ── Phase 2: Collapse Weight Sweep ────────────────────────────
echo ""
echo ">>> PHASE 2: collapse_weight sweep (dw=$BEST_DW)"
python grid_search_cdmoe.py \
    --base-ckpt $BASE_CKPT \
    --data-path $DATA_PATH \
    --phase 2 \
    --best-diversity-weight $BEST_DW \
    --train-steps $STEPS

# Extract best collapse_weight
BEST_CW=$(python -c "
import json
with open('grid_search_results/phase2_summary.json') as f:
    results = json.load(f)
best = max(results, key=lambda x: x['summary']['composite_score'])
print(best['config']['collapse_weight'])
")
echo ""
echo ">>> Phase 2 BEST collapse_weight = $BEST_CW"

# ── Phase 3: Gate Temperature Sweep ────────────────────────────
echo ""
echo ">>> PHASE 3: gate_temp sweep (dw=$BEST_DW, cw=$BEST_CW)"
python grid_search_cdmoe.py \
    --base-ckpt $BASE_CKPT \
    --data-path $DATA_PATH \
    --phase 3 \
    --best-diversity-weight $BEST_DW \
    --best-collapse-weight $BEST_CW \
    --train-steps $STEPS

# Extract best gate_temp
BEST_GT=$(python -c "
import json
with open('grid_search_results/phase3_summary.json') as f:
    results = json.load(f)
best = max(results, key=lambda x: x['summary']['composite_score'])
print(best['config']['gate_temp'])
")
echo ""
echo ">>> Phase 3 BEST gate_temp = $BEST_GT"

# ── Full Training with Best Config ─────────────────────────────
echo ""
echo "============================================================"
echo "  FULL TRAINING WITH BEST CONFIG"
echo "  diversity_weight=$BEST_DW"
echo "  collapse_weight=$BEST_CW"
echo "  gate_temp=$BEST_GT"
echo "  Started: $(date)"
echo "============================================================"

python finetune_concept_gate.py \
    --data-path $DATA_PATH \
    --ckpt $BASE_CKPT \
    --mode loss_only \
    --diversity-weight $BEST_DW \
    --collapse-weight $BEST_CW \
    --gate-temp $BEST_GT \
    --min-expert-util 0.5 \
    --epochs 1 --batch-size 16 --lr 3e-4 \
    --ckpt-every 20000

# ── Final Analysis ─────────────────────────────────────────────
echo ""
echo ">>> Running final analysis..."

# Find the latest checkpoint
FINAL_CKPT=$(find results-cdmoe/ -name "final.pt" -newer grid_search_results/ | sort | tail -1)
echo "  Final checkpoint: $FINAL_CKPT"

python analyze_cdmoe_result.py \
    --base-ckpt $BASE_CKPT \
    --cdmoe-ckpt $FINAL_CKPT \
    --output-dir concept_analysis/cdmoe_best

echo ""
echo "============================================================"
echo "  ALL DONE!"
echo "  Best config: dw=$BEST_DW, cw=$BEST_CW, gt=$BEST_GT"
echo "  Results in: concept_analysis/cdmoe_best/"
echo "  Finished: $(date)"
echo "============================================================"
