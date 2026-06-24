#!/bin/bash
# =============================================================================
# setup_env.sh — One-time environment setup for CD-MoE project
#
# Run once on a new machine or after conda env is deleted.
# Usage: bash scripts/setup_env.sh
# =============================================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "=== CD-MoE Project Setup ==="
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# ── 1. Create conda environment ──────────────────────────────────────────────
if conda env list | grep -q "^dit "; then
    echo "[✓] Conda env 'dit' already exists. Skipping creation."
else
    echo "[+] Creating conda env from environment.yml..."
    conda env create -f environment.yml
    echo "[✓] Conda env 'dit' created."
fi

# ── 2. Download pretrained DiT-XL/2 checkpoint ───────────────────────────────
CKPT_PATH="$PROJECT_ROOT/pretrained_models/DiT-XL-2-256x256.pt"
if [ -f "$CKPT_PATH" ]; then
    echo "[✓] Pretrained checkpoint already exists: $CKPT_PATH"
else
    echo "[+] Downloading DiT-XL/2-256x256 pretrained checkpoint..."
    mkdir -p "$PROJECT_ROOT/pretrained_models"
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt" \
        -O "$CKPT_PATH"
    echo "[✓] Downloaded to $CKPT_PATH"
fi

# ── 3. Make all scripts executable ───────────────────────────────────────────
echo "[+] Making all scripts executable..."
chmod +x "$PROJECT_ROOT/scripts/"*.sh
chmod +x "$PROJECT_ROOT/training_watchdog.sh"
chmod +x "$PROJECT_ROOT/auto_cleanup_watcher.sh"
echo "[✓] All scripts are now executable."

# ── 4. Create output directories ─────────────────────────────────────────────
mkdir -p "$PROJECT_ROOT/routing_analysis"
mkdir -p "$PROJECT_ROOT/results-finetune"
mkdir -p "$PROJECT_ROOT/ablation_results"
echo "[✓] Output directories created."

echo ""
echo "=== Setup Complete ==="
echo "Activate with:  conda activate dit"
echo "Run training:   bash scripts/run_training.sh"
echo "Run sampling:   bash scripts/run_sampling.sh --ckpt <path>"
