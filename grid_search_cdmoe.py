"""
grid_search_cdmoe.py — Automated hyperparameter grid search for CD-MoE.

Runs multiple short training runs (10k steps each) with different hyperparameters,
auto-analyzes each, and produces a summary comparison.

Usage:
    python grid_search_cdmoe.py \
        --base-ckpt results-finetune/004-DiT-XL-2-MoE-finetune-moe24-27/checkpoints/0115000.pt \
        --data-path /local/a/imagenet/imagenet2012/train/ \
        --phase 1

    # After Phase 1, check results and run Phase 2 with best diversity_weight:
    python grid_search_cdmoe.py \
        --base-ckpt results-finetune/004-DiT-XL-2-MoE-finetune-moe24-27/checkpoints/0115000.pt \
        --data-path /local/a/imagenet/imagenet2012/train/ \
        --phase 2 --best-diversity-weight 1.0

    # Phase 3:
    python grid_search_cdmoe.py ... --phase 3 --best-diversity-weight 1.0 --best-collapse-weight 1.0
"""

import subprocess
import json
import os
import sys
import argparse
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRID_RESULTS_DIR = os.path.join(BASE_DIR, "grid_search_results")


def run_single_config(config, args):
    """Train a single config for short steps, then analyze."""
    run_name = config["name"]
    run_dir = os.path.join(GRID_RESULTS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Running: {run_name}")
    print(f"  diversity_weight={config['diversity_weight']}, "
          f"collapse_weight={config['collapse_weight']}, "
          f"gate_temp={config['gate_temp']}")
    print(f"{'='*60}")

    # Step 1: Train (short run using --max-steps)
    train_log = os.path.join(run_dir, "train.log")
    train_cmd = [
        sys.executable, os.path.join(BASE_DIR, "finetune_concept_gate.py"),
        "--data-path", args.data_path,
        "--ckpt", args.base_ckpt,
        "--mode", "loss_only",
        "--diversity-weight", str(config["diversity_weight"]),
        "--min-expert-util", str(config["min_expert_util"]),
        "--collapse-weight", str(config["collapse_weight"]),
        "--gate-temp", str(config["gate_temp"]),
        "--epochs", "1",
        "--batch-size", "16",
        "--lr", str(config.get("lr", 3e-4)),
        "--max-steps", str(args.train_steps),
        "--ckpt-every", str(args.train_steps + 1),  # don't save intermediate
        "--log-every", "100",
        "--results-dir", os.path.join(run_dir, "results-cdmoe"),
    ]

    print(f"  Training for {args.train_steps} steps...")
    with open(train_log, "w") as log_f:
        subprocess.run(train_cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=BASE_DIR)

    # Find the checkpoint
    results_subdir = os.path.join(run_dir, "results-cdmoe")
    ckpt_path = None
    if os.path.exists(results_subdir):
        for d in sorted(os.listdir(results_subdir)):
            ckpt_dir = os.path.join(results_subdir, d, "checkpoints")
            if os.path.exists(ckpt_dir):
                # Use final.pt if exists, else latest numbered
                final = os.path.join(ckpt_dir, "final.pt")
                if os.path.exists(final):
                    ckpt_path = final
                else:
                    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
                    if ckpts:
                        ckpt_path = os.path.join(ckpt_dir, ckpts[-1])

    if ckpt_path is None:
        print(f"  WARNING: No checkpoint found for {run_name}")
        return None

    print(f"  Checkpoint: {ckpt_path}")

    # Step 2: Analyze
    metrics_path = os.path.join(run_dir, "metrics.json")
    analyze_cmd = [
        sys.executable, os.path.join(BASE_DIR, "quick_analyze.py"),
        "--base-ckpt", args.base_ckpt,
        "--cdmoe-ckpt", ckpt_path,
        "--output", metrics_path,
        "--num-timesteps", "30",  # fewer timesteps for speed
    ]

    print(f"  Analyzing routing...")
    analyze_log = os.path.join(run_dir, "analyze.log")
    with open(analyze_log, "w") as log_f:
        subprocess.run(analyze_cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=BASE_DIR)

    # Load metrics
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        metrics["config"] = config
        # Re-save with config
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
    else:
        print(f"  WARNING: Analysis failed for {run_name}")
        return None


def print_comparison(all_results):
    """Print a comparison table of all results."""
    print(f"\n{'='*100}")
    print(f"  GRID SEARCH RESULTS COMPARISON")
    print(f"{'='*100}")
    print(f"{'Config':<30} {'Unique/32':>10} {'AvgMaxUtil':>10} {'MinUtil':>10} "
          f"{'JSD_early':>10} {'JSD_ratio':>10} {'Score':>10}")
    print(f"{'─'*100}")

    # Sort by composite score
    sorted_results = sorted(all_results, key=lambda x: x["summary"]["composite_score"], reverse=True)

    for r in sorted_results:
        c = r["config"]
        s = r["summary"]
        name = c["name"]
        print(f"{name:<30} {s['total_unique_dominant']:>10} {s['avg_max_util']:>10.1%} "
              f"{s['worst_min_util']:>10.1%} {s['avg_jsd_early']:>10.6f} "
              f"{s['avg_jsd_ratio']:>10.1f}× {s['composite_score']:>10.2f}")

    print(f"{'='*100}")
    best = sorted_results[0]
    print(f"\n  🏆 BEST: {best['config']['name']} (score={best['summary']['composite_score']:.2f})")
    print(f"     diversity_weight={best['config']['diversity_weight']}, "
          f"collapse_weight={best['config']['collapse_weight']}, "
          f"gate_temp={best['config']['gate_temp']}")


def main(args):
    os.makedirs(GRID_RESULTS_DIR, exist_ok=True)

    configs = []

    if args.phase == 1:
        # Phase 1: Sweep diversity_weight
        for dw in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
            configs.append({
                "name": f"p1_dw{dw}",
                "diversity_weight": dw,
                "collapse_weight": 1.0,
                "min_expert_util": 0.5,
                "gate_temp": 1.0,
            })

    elif args.phase == 2:
        # Phase 2: Sweep collapse_weight with best diversity_weight
        dw = args.best_diversity_weight
        for cw in [0.0, 0.5, 1.0, 2.0, 5.0]:
            configs.append({
                "name": f"p2_dw{dw}_cw{cw}",
                "diversity_weight": dw,
                "collapse_weight": cw,
                "min_expert_util": 0.5,
                "gate_temp": 1.0,
            })

    elif args.phase == 3:
        # Phase 3: Sweep gate temperature
        dw = args.best_diversity_weight
        cw = args.best_collapse_weight
        for gt in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            configs.append({
                "name": f"p3_dw{dw}_cw{cw}_gt{gt}",
                "diversity_weight": dw,
                "collapse_weight": cw,
                "min_expert_util": 0.5,
                "gate_temp": gt,
            })

    elif args.phase == 0:
        # Custom: run specific configs
        if args.custom_configs:
            configs = json.loads(args.custom_configs)
        else:
            print("Phase 0 requires --custom-configs as JSON string")
            return

    print(f"\n{'#'*60}")
    print(f"  CD-MoE GRID SEARCH — Phase {args.phase}")
    print(f"  {len(configs)} configurations, {args.train_steps} steps each")
    print(f"  Estimated time: ~{len(configs) * (args.train_steps / 60 * 10 / 60 + 5):.0f} minutes")
    print(f"{'#'*60}")

    all_results = []

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] ", end="")
        result = run_single_config(config, args)
        if result:
            all_results.append(result)

    # Print comparison
    if all_results:
        print_comparison(all_results)

        # Save overall summary
        summary_path = os.path.join(GRID_RESULTS_DIR, f"phase{args.phase}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--phase", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--train-steps", type=int, default=10000,
                        help="Steps per config (default: 10000 for grid search)")
    parser.add_argument("--best-diversity-weight", type=float, default=0.5)
    parser.add_argument("--best-collapse-weight", type=float, default=1.0)
    parser.add_argument("--custom-configs", type=str, default=None)
    args = parser.parse_args()
    main(args)
