#!/bin/bash
# training_watchdog.sh
#
# Monitors whether finetune_moe.py is still running. If it dies:
#   1. Tries to email you immediately (so you know within minutes, not hours)
#   2. Automatically finds the latest VALID checkpoint and resumes training
#      from it -- so you don't lose hours of idle GPU time waiting for you
#      to wake up and notice.
#
# USAGE:
#   chmod +x training_watchdog.sh
#   nohup ./training_watchdog.sh your_email@purdue.edu > watchdog.log 2>&1 &
#   disown -h
#
# TO STOP:
#   pkill -f training_watchdog.sh

EMAIL="$1"
CHECK_INTERVAL=120   # check every 2 minutes
RESULTS_DIR="results-finetune"
DATA_PATH="/local/a/imagenet/imagenet2012/train/"
CKPT="pretrained_models/DiT-XL-2-256x256.pt"
MOE_BLOCKS="24 25 26 27"
NUM_EXPERTS=4
EPOCHS=1
BATCH_SIZE=16
LR=1e-4
LOG_EVERY=100
CKPT_EVERY=5000
MIN_VALID_SIZE_BYTES=1000000000   # 1GB -- below this, a checkpoint is corrupted

if [ -z "$EMAIL" ]; then
    echo "Usage: $0 <your_email@purdue.edu>"
    exit 1
fi

echo "Starting training watchdog."
echo "  Alert email: $EMAIL"
echo "  Check interval: ${CHECK_INTERVAL}s"
echo "  Started at: $(date)"
echo "----------------------------------------"

send_alert() {
    local subject="$1"
    local body="$2"
    echo "$body" | mail -s "$subject" "$EMAIL" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  -> Alert email sent successfully."
    else
        echo "  -> WARNING: failed to send email (mail command may not be configured on this server)."
    fi
}

find_latest_valid_checkpoint() {
    # Search all experiment folders for checkpoints, newest step first,
    # and actually TRY LOADING each one with torch.load until one
    # succeeds. Size alone is NOT a reliable validity check -- a file
    # can be exactly the right size on disk and still be corrupted
    # (e.g. if writes got reordered/interrupted during a disk-full
    # event), as happened on 2026-06-21 (steps 20000, 25000, 35000 were
    # all full-size but corrupted; only 30000 actually loaded).
    candidates=$(find "$RESULTS_DIR" -name "[0-9]*.pt" -type f -size +${MIN_VALID_SIZE_BYTES}c 2>/dev/null \
        | while read -r f; do
            step=$(basename "$f" .pt)
            echo "$step $f"
          done | sort -rn)  # newest step first

    while read -r step fpath; do
        [ -z "$fpath" ] && continue
        echo "  Checking candidate: $fpath (step $step)..." >&2
        if python -c "
import torch
import sys
try:
    ckpt = torch.load('$fpath', map_location='cpu', weights_only=False)
    assert 'train_steps' in ckpt
    sys.exit(0)
except Exception as e:
    print(f'    -> failed: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&2; then
            echo "  -> VALID: $fpath" >&2
            echo "$fpath"
            return 0
        else
            echo "  -> CORRUPTED, trying older checkpoint..." >&2
        fi
    done <<< "$candidates"

    return 1
}

restart_training() {
    local resume_ckpt
    resume_ckpt=$(find_latest_valid_checkpoint)

    if [ -z "$resume_ckpt" ] || [ ! -f "$resume_ckpt" ]; then
        echo "  -> ERROR: no valid (loadable) checkpoint found to resume from. NOT auto-restarting."
        send_alert "[Diff-MoE WATCHDOG] Training died, NO valid checkpoint found" \
            "Training process died and every checkpoint I tried failed to load. Manual intervention needed. Check the server now."
        return 1
    fi

    echo "  -> Resuming from: $resume_ckpt"

    nohup python finetune_moe.py \
        --data-path "$DATA_PATH" \
        --ckpt "$CKPT" \
        --moe-blocks $MOE_BLOCKS \
        --num-experts "$NUM_EXPERTS" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --log-every "$LOG_EVERY" \
        --ckpt-every "$CKPT_EVERY" \
        --resume "$resume_ckpt" \
        > "training_auto_resume_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

    disown -h
    NEW_PID=$!
    echo "  -> Restarted training, new PID: $NEW_PID"

    send_alert "[Diff-MoE WATCHDOG] Training died, auto-restarted from $resume_ckpt" \
        "Your training process died, but I automatically restarted it from checkpoint: $resume_ckpt
New PID: $NEW_PID
Time: $(date)

Check on it when you can, but it should already be running again."
}

MAX_CONSECUTIVE_RESTARTS=3   # after this many rapid restarts, stop and alert loudly instead of looping
RESTART_COOLDOWN_SECONDS=600 # if we hit the limit, wait this long before trying again
consecutive_restarts=0

while true; do
    if pgrep -f "python finetune_moe.py" > /dev/null; then
        consecutive_restarts=0  # reset counter once training is confirmed healthy
        :
    else
        echo ""
        echo "=== $(date): Training process NOT FOUND (died) ==="

        if [ "$consecutive_restarts" -ge "$MAX_CONSECUTIVE_RESTARTS" ]; then
            echo "  -> Hit $MAX_CONSECUTIVE_RESTARTS consecutive restarts. Something is "
            echo "     repeatedly broken (e.g. /tmp full again, disk quota again)."
            echo "     Pausing for ${RESTART_COOLDOWN_SECONDS}s instead of looping forever."
            send_alert "[Diff-MoE WATCHDOG] CRASH LOOP -- needs manual attention" \
                "Training has crashed $MAX_CONSECUTIVE_RESTARTS times in a row immediately after restart. This usually means something is broken at a deeper level (disk full again, /tmp full again, etc.) rather than a one-off failure. I've stopped auto-restarting to avoid spamming you. Please check the server manually: df -h /tmp, df -h ~, quota -s, and the latest training_auto_resume_*.log file."
            sleep "$RESTART_COOLDOWN_SECONDS"
            consecutive_restarts=0
            continue
        fi

        restart_training
        consecutive_restarts=$((consecutive_restarts + 1))
    fi
    sleep "$CHECK_INTERVAL"
done