"""
Checkpoint cleanup script -- run this SEPARATELY from your training run
(in a second terminal/tmux pane), NOT inside the training process.

It deletes old checkpoints in a results-finetune experiment folder,
keeping only the N most recent ones. This prevents checkpoints from
filling your disk quota during a long run.

SAFE BY DESIGN:
    - Never touches the training process itself.
    - Only deletes .pt files matching the step-number pattern
      (e.g. 0005000.pt), never touches log.txt or other files.
    - Always keeps the most recent N checkpoints (default 2), so you
      always have a safe resume point even if something goes wrong
      right after a delete.
    - Skips (does not delete) any file currently being written --
      checked via file modification time being very recent (last 60s),
      to avoid deleting a checkpoint mid-write.
    - Dry-run by default -- shows what WOULD be deleted without
      actually deleting, until you pass --confirm.

USAGE:
    # See what would be deleted (safe, no changes made):
    python cleanup_checkpoints.py --dir results-finetune/003-DiT-XL-2-MoE-finetune-moe24-27/checkpoints

    # Actually delete, keeping the 2 most recent:
    python cleanup_checkpoints.py --dir results-finetune/003-DiT-XL-2-MoE-finetune-moe24-27/checkpoints --confirm

    # Keep 3 most recent instead of the default 2:
    python cleanup_checkpoints.py --dir results-finetune/003-DiT-XL-2-MoE-finetune-moe24-27/checkpoints --keep 3 --confirm
"""

import argparse
import os
import re
import time


def find_checkpoints(checkpoint_dir, min_valid_size_gb=1.0):
    """
    Find all step-numbered checkpoint files (e.g. 0005000.pt), excluding
    final.pt (which we always keep -- it's your end-of-run result).

    IMPORTANT: also flags checkpoints smaller than `min_valid_size_gb` as
    LIKELY CORRUPTED (a partial/failed write), since a real DiT-XL/2
    checkpoint is normally several GB. A suspiciously small file is
    almost certainly a crash mid-save, not a real checkpoint -- treat it
    as junk to delete, never as something to "keep" just because it's
    the most recent by step number.

    Returns list of (step_number, filepath, size_bytes, mtime, is_valid_size)
    sorted by step number ascending (oldest first).
    """
    pattern = re.compile(r"^(\d+)\.pt$")
    checkpoints = []
    min_valid_bytes = min_valid_size_gb * 1e9

    for fname in os.listdir(checkpoint_dir):
        match = pattern.match(fname)
        if match:
            step = int(match.group(1))
            fpath = os.path.join(checkpoint_dir, fname)
            size = os.path.getsize(fpath)
            mtime = os.path.getmtime(fpath)
            is_valid_size = size >= min_valid_bytes
            checkpoints.append((step, fpath, size, mtime, is_valid_size))

    checkpoints.sort(key=lambda x: x[0])  # sort by step number, ascending
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Clean up old training checkpoints, keeping the N most recent.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the checkpoints/ folder.")
    parser.add_argument("--keep", type=int, default=2, help="Number of most recent VALID checkpoints to keep (default: 2).")
    parser.add_argument("--confirm", action="store_true", help="Actually delete files. Without this flag, it's a dry run.")
    parser.add_argument("--min-age-seconds", type=int, default=60,
                        help="Skip files modified more recently than this many seconds ago, "
                             "to avoid deleting a checkpoint that's still being written (default: 60).")
    parser.add_argument("--min-valid-size-gb", type=float, default=1.0,
                        help="Checkpoints smaller than this are treated as corrupted/partial "
                             "writes and are ALWAYS eligible for deletion, regardless of "
                             "recency or --keep (default: 1.0 GB).")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"ERROR: directory does not exist: {args.dir}")
        return

    checkpoints = find_checkpoints(args.dir, min_valid_size_gb=args.min_valid_size_gb)

    if len(checkpoints) == 0:
        print(f"No step-numbered checkpoints found in {args.dir}")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) in {args.dir}:")
    total_size_gb = 0
    now = time.time()
    for step, fpath, size, mtime, is_valid_size in checkpoints:
        age_sec = now - mtime
        size_gb = size / 1e9
        total_size_gb += size_gb
        recent_flag = " (RECENT -- will skip, possibly still writing)" if age_sec < args.min_age_seconds else ""
        corrupt_flag = " [SUSPICIOUSLY SMALL -- likely corrupted/partial write]" if not is_valid_size else ""
        print(f"  step {step:>8} | {size_gb:6.2f} GB | modified {age_sec:.0f}s ago{recent_flag}{corrupt_flag}")
    print(f"\nTotal size: {total_size_gb:.2f} GB")

    # Split into valid-size and corrupted-size checkpoints.
    # Corrupted ones are ALWAYS candidates for deletion (subject only to
    # the recency safety check), regardless of step number or --keep --
    # we never want to "keep" a broken file just because it's the newest.
    valid_checkpoints = [c for c in checkpoints if c[4]]
    corrupted_checkpoints = [c for c in checkpoints if not c[4]]

    if corrupted_checkpoints:
        print(f"\n{len(corrupted_checkpoints)} checkpoint(s) flagged as corrupted/partial "
              f"(under {args.min_valid_size_gb} GB) -- these are NEVER kept:")
        for step, fpath, size, mtime, _ in corrupted_checkpoints:
            print(f"  step {step} -- {size/1e9:.3f} GB -- {fpath}")

    # Among VALID checkpoints only, keep the N most recent by step number.
    to_keep = set(c[0] for c in valid_checkpoints[-args.keep:]) if args.keep > 0 else set()
    to_delete = corrupted_checkpoints + [c for c in valid_checkpoints if c[0] not in to_keep]

    # Safety: never delete anything modified too recently (might still be writing),
    # EVEN IF it's flagged as corrupted -- it might just be mid-write right now.
    safe_to_delete = [c for c in to_delete if (now - c[3]) >= args.min_age_seconds]
    skipped_recent = [c for c in to_delete if (now - c[3]) < args.min_age_seconds]

    if skipped_recent:
        print(f"\nSkipping {len(skipped_recent)} checkpoint(s) modified in the last "
              f"{args.min_age_seconds}s (might still be writing):")
        for step, fpath, size, mtime, _ in skipped_recent:
            print(f"  step {step} -- {fpath}")

    if not safe_to_delete:
        print(f"\nNothing to delete right now. Keeping {len(checkpoints) - len(skipped_recent)} checkpoint(s).")
        return

    freed_gb = sum(c[2] for c in safe_to_delete) / 1e9
    print(f"\n{'Would delete' if not args.confirm else 'Deleting'} "
          f"{len(safe_to_delete)} checkpoint(s), freeing {freed_gb:.2f} GB:")
    for step, fpath, size, mtime, is_valid_size in safe_to_delete:
        reason = "corrupted/partial" if not is_valid_size else "old (keeping newer ones)"
        print(f"  step {step:>8} | {size/1e9:.2f} GB | {reason} | {fpath}")

    if not args.confirm:
        print("\nDRY RUN -- no files were deleted. Re-run with --confirm to actually delete.")
    else:
        for step, fpath, size, mtime, _ in safe_to_delete:
            os.remove(fpath)
        print(f"\nDeleted {len(safe_to_delete)} checkpoint(s). Freed {freed_gb:.2f} GB.")
        print(f"Kept: {sorted(to_keep)}")


if __name__ == "__main__":
    main()