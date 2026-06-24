#!/bin/bash
# =============================================================================
# check_status.sh — One command to see everything at a glance
#
# Usage: bash scripts/check_status.sh
# =============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "╔══════════════════════════════════════════════════════╗"
echo "║           CD-MoE Project Status Check               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── GPU Status ────────────────────────────────────────────────────────────────
echo "━━━ GPU Status ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' '{printf "  GPU %s (%s): %s%% util | %s/%s MB\n", $1,$2,$3,$4,$5}' || \
    echo "  nvidia-smi not available"
echo ""

# ── Training Process ─────────────────────────────────────────────────────────
echo "━━━ Training Process ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
TRAIN_PIDS=$(pgrep -f "finetune_moe.py" 2>/dev/null)
if [ -n "$TRAIN_PIDS" ]; then
    echo "  [RUNNING] PID(s): $TRAIN_PIDS"
else
    echo "  [STOPPED] No training process found"
fi

WATCHDOG_PIDS=$(pgrep -f "training_watchdog.sh" 2>/dev/null)
if [ -n "$WATCHDOG_PIDS" ]; then
    echo "  [WATCHDOG RUNNING] PID(s): $WATCHDOG_PIDS"
else
    echo "  [WATCHDOG STOPPED]"
fi
echo ""

# ── Latest Training Log ───────────────────────────────────────────────────────
echo "━━━ Latest Training Progress ━━━━━━━━━━━━━━━━━━━━━━━━━"
LATEST_LOG=$(ls -t training_auto_resume_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "  Log: $LATEST_LOG"
    grep "Train Loss" "$LATEST_LOG" | tail -3 | \
        awk '{printf "  %s\n", $0}'
else
    echo "  No training logs found"
fi
echo ""

# ── Checkpoints ───────────────────────────────────────────────────────────────
echo "━━━ Checkpoints ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CKPT_DIR="results-finetune/004-DiT-XL-2-MoE-finetune-moe24-27/checkpoints"
if [ -d "$CKPT_DIR" ]; then
    ls -lh "$CKPT_DIR/"*.pt 2>/dev/null | \
        awk '{printf "  %-15s %s  %s\n", $5, $6" "$7, $9}' || \
        echo "  No checkpoints found"
else
    echo "  Checkpoint directory not found: $CKPT_DIR"
fi

TMP_CKPTS=$(ls -lh /tmp/*.pt 2>/dev/null | wc -l)
if [ "$TMP_CKPTS" -gt 0 ]; then
    echo ""
    echo "  /tmp checkpoints (not yet moved to NFS):"
    ls -lh /tmp/*.pt 2>/dev/null | awk '{printf "    %-12s %s\n", $5, $9}'
fi
echo ""

# ── Disk Quota ────────────────────────────────────────────────────────────────
echo "━━━ Disk Quota ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
quota -s 2>/dev/null | grep -v "^Disk" | grep -v "^$" | grep -v "Filesystem" | \
    awk '{
        used=$2; quota=$3; limit=$4;
        gsub(/M/,"",used); gsub(/M/,"",quota); gsub(/M/,"",limit);
        pct=int(used/limit*100);
        bar=""; for(i=0;i<pct/5;i++) bar=bar"█";
        printf "  Used: %sM / %sM (%d%%)\n  [%-20s]\n", used, limit, pct, bar
    }' || echo "  quota command unavailable"

echo ""
echo "  /tmp (local scratch):   $(df -h /tmp | tail -1 | awk '{print $3"/"$2" ("$5" used)"}')"
echo ""

# ── Results Summary ───────────────────────────────────────────────────────────
echo "━━━ Results ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Sample images:"
ls -lh sample_finetuned_*.png sample_diffmoe*.png 2>/dev/null | \
    awk '{printf "    %-12s %s\n", $5, $9}' || echo "    None"
echo ""
echo "  Routing analysis dirs:"
ls -d routing_analysis*/ 2>/dev/null | sed 's/^/    /' || echo "    None"
echo ""
echo "  Ablation results:"
ls -d ablation_results/*/ 2>/dev/null | sed 's/^/    /' || echo "    None yet"
echo ""
