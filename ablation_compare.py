"""
ablation_compare.py
===================
Compare routing analysis results across the E1-E4 ablation study.

Reads all routing analysis output directories under ablation_results/ and
produces a single multi-panel comparison figure showing:
  - Expert utilization per num_experts config
  - Router entropy: pretrained vs finetuned (4 experts)
  - Load balance (CV) across all configs
  - Temporal routing comparison: pretrained 4exp vs finetuned 4exp

Usage:
    python ablation_compare.py
    python ablation_compare.py --results-dir ablation_results --output ablation_comparison.png
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json


# ── Helper: load routing stats from a run directory ──────────────────────────

def load_stats(run_dir: str):
    """
    Load routing statistics saved by analyze_routing.py.
    Returns a dict with utilization, entropy, CV, temporal data.
    Returns None if directory is missing or incomplete.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None

    stats_file = run_dir / "routing_stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            return json.load(f)

    # Fallback: try to parse from the text log
    log_file = run_dir / "run.log"
    if not log_file.exists():
        return None

    stats = {"utilization": {}, "entropy": {}, "cv": {}, "label": run_dir.name}
    with open(log_file) as f:
        lines = f.readlines()

    current_block = None
    for line in lines:
        line = line.strip()
        if "Block" in line and "Expert utilization:" not in line and ":" in line:
            try:
                current_block = int(line.split("Block")[1].split(":")[0].strip())
                stats["utilization"][current_block] = {}
            except Exception:
                pass
        if "Expert utilization:" in line and current_block is not None:
            parts = line.split("Expert utilization:")[1].strip().split(",")
            for p in parts:
                p = p.strip()
                if "=" in p:
                    k, v = p.split("=")
                    exp_idx = int(k.strip().replace("E", ""))
                    val = float(v.strip().replace("%", "")) / 100.0
                    stats["utilization"][current_block][exp_idx] = val
        if "Avg router entropy:" in line and current_block is not None:
            try:
                entropy_str = line.split("Avg router entropy:")[1].split("/")[0].strip()
                if current_block not in stats["entropy"]:
                    stats["entropy"][current_block] = []
                stats["entropy"][current_block] = float(entropy_str)
            except Exception:
                pass
        if "Load balance CV:" in line and current_block is not None:
            try:
                cv_str = line.split("Load balance CV:")[1].split("(")[0].strip()
                stats["cv"][current_block] = float(cv_str)
            except Exception:
                pass

    return stats if stats["utilization"] else None


# ── Main comparison figure ────────────────────────────────────────────────────

def make_comparison_figure(results_dir: str, output_path: str):
    results_dir = Path(results_dir)

    # Expected run directories from run_ablation.sh
    configs = {
        "pretrained_1exp_mixed":   {"n_exp": 1, "model": "pretrained", "cat": "mixed"},
        "pretrained_1exp_single":  {"n_exp": 1, "model": "pretrained", "cat": "single"},
        "pretrained_2exp_mixed":   {"n_exp": 2, "model": "pretrained", "cat": "mixed"},
        "pretrained_2exp_single":  {"n_exp": 2, "model": "pretrained", "cat": "single"},
        "pretrained_3exp_mixed":   {"n_exp": 3, "model": "pretrained", "cat": "mixed"},
        "pretrained_3exp_single":  {"n_exp": 3, "model": "pretrained", "cat": "single"},
        "pretrained_4exp_mixed":   {"n_exp": 4, "model": "pretrained", "cat": "mixed"},
        "pretrained_4exp_single":  {"n_exp": 4, "model": "pretrained", "cat": "single"},
        "finetuned_4exp_mixed":    {"n_exp": 4, "model": "finetuned",  "cat": "mixed"},
        "finetuned_4exp_single":   {"n_exp": 4, "model": "finetuned",  "cat": "single"},
    }

    # Load all available stats
    loaded = {}
    for name, meta in configs.items():
        stats = load_stats(results_dir / name)
        if stats is not None:
            stats.update(meta)
            loaded[name] = stats
        else:
            print(f"  [!] Missing or incomplete: {name}")

    if not loaded:
        print("No results found. Run scripts/run_ablation.sh first.")
        return

    print(f"\nLoaded {len(loaded)}/{len(configs)} runs.")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("CD-MoE Ablation Study: Expert Count × Model Type × Category",
                 fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Plot 1: Load balance (CV) vs num_experts — pretrained, mixed ─────────
    ax1 = fig.add_subplot(gs[0, 0])
    exp_counts = [1, 2, 3, 4]
    cvs_mixed = []
    cvs_single = []
    for n in exp_counts:
        key_m = f"pretrained_{n}exp_mixed"
        key_s = f"pretrained_{n}exp_single"
        # average CV across all blocks
        if key_m in loaded and loaded[key_m]["cv"]:
            cvs_mixed.append(np.mean(list(loaded[key_m]["cv"].values())))
        else:
            cvs_mixed.append(float('nan'))
        if key_s in loaded and loaded[key_s]["cv"]:
            cvs_single.append(np.mean(list(loaded[key_s]["cv"].values())))
        else:
            cvs_single.append(float('nan'))

    x = np.arange(len(exp_counts))
    w = 0.35
    ax1.bar(x - w/2, cvs_mixed, w, label='Mixed categories', color='steelblue', alpha=0.8)
    ax1.bar(x + w/2, cvs_single, w, label='Single category', color='coral', alpha=0.8)
    ax1.set_xlabel('Number of Experts')
    ax1.set_ylabel('Load Imbalance (CV)\n← lower = more balanced')
    ax1.set_title('E1–E4: Load Balance\n(Pretrained model)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'E{n}\n({n} exp)' for n in exp_counts])
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # ── Plot 2: Router entropy vs num_experts — pretrained, mixed ────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ent_mixed = []
    ent_single = []
    for n in exp_counts:
        key_m = f"pretrained_{n}exp_mixed"
        key_s = f"pretrained_{n}exp_single"
        max_ent = np.log(n) if n > 1 else 0.001  # theoretical max
        if key_m in loaded and loaded[key_m]["entropy"]:
            avg = np.mean(list(loaded[key_m]["entropy"].values()))
            ent_mixed.append(avg / max_ent * 100)  # as % of max
        else:
            ent_mixed.append(float('nan'))
        if key_s in loaded and loaded[key_s]["entropy"]:
            avg = np.mean(list(loaded[key_s]["entropy"].values()))
            ent_single.append(avg / max_ent * 100)
        else:
            ent_single.append(float('nan'))

    ax2.bar(x - w/2, ent_mixed, w, label='Mixed categories', color='steelblue', alpha=0.8)
    ax2.bar(x + w/2, ent_single, w, label='Single category', color='coral', alpha=0.8)
    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Max (uniform)')
    ax2.set_xlabel('Number of Experts')
    ax2.set_ylabel('Router Entropy\n(% of theoretical max)')
    ax2.set_title('E1–E4: Router Entropy\n(Pretrained model)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'E{n}\n({n} exp)' for n in exp_counts])
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 115)
    ax2.grid(axis='y', alpha=0.3)

    # ── Plot 3: Pretrained vs Finetuned comparison (4 experts) ───────────────
    ax3 = fig.add_subplot(gs[0, 2])
    model_labels = ['Pretrained\n(mixed)', 'Pretrained\n(single)',
                    'Finetuned\n(mixed)', 'Finetuned\n(single)']
    model_keys = ['pretrained_4exp_mixed', 'pretrained_4exp_single',
                  'finetuned_4exp_mixed', 'finetuned_4exp_single']
    model_cvs = []
    for key in model_keys:
        if key in loaded and loaded[key]["cv"]:
            model_cvs.append(np.mean(list(loaded[key]["cv"].values())))
        else:
            model_cvs.append(float('nan'))

    colors = ['#5b9bd5', '#9dc3e6', '#e05c2a', '#f4a460']
    bars = ax3.bar(range(4), model_cvs, color=colors, alpha=0.85, edgecolor='white')
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Load Imbalance (CV)\n← lower = more balanced')
    ax3.set_title('Pretrained vs Finetuned\n(4 experts, Q1 from meeting)')
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(model_labels, fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, model_cvs):
        if not np.isnan(val):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # ── Plot 4–7: Expert utilization heatmaps for each config ────────────────
    heatmap_keys = ['pretrained_4exp_mixed', 'pretrained_4exp_single',
                    'finetuned_4exp_mixed', 'finetuned_4exp_single']
    heatmap_titles = ['Pretrained — Mixed', 'Pretrained — Single category',
                      'Finetuned — Mixed', 'Finetuned — Single category']
    heatmap_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    heatmap_axes.append(fig.add_subplot(gs[2, 0]))

    for ax, key, title in zip(heatmap_axes, heatmap_keys, heatmap_titles):
        if key not in loaded or not loaded[key]["utilization"]:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            continue
        util = loaded[key]["utilization"]
        blocks = sorted(util.keys())
        n_exp = loaded[key]["n_exp"]
        matrix = np.zeros((len(blocks), n_exp))
        for i, b in enumerate(blocks):
            for e in range(n_exp):
                matrix[i, e] = util[b].get(e, 0.0)

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)
        ax.set_xticks(range(n_exp))
        ax.set_xticklabels([f'E{e}' for e in range(n_exp)], fontsize=8)
        ax.set_yticks(range(len(blocks)))
        ax.set_yticklabels([f'Block {b}' for b in blocks], fontsize=8)
        ax.set_xlabel('Expert')
        ax.set_title(title, fontsize=9, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(len(blocks)):
            for j in range(n_exp):
                ax.text(j, i, f'{matrix[i, j]:.2f}',
                        ha='center', va='center', fontsize=7,
                        color='white' if matrix[i, j] > 0.35 else 'black')

    # ── Plot 8: Summary text — key findings ───────────────────────────────────
    ax_text = fig.add_subplot(gs[2, 1:])
    ax_text.axis('off')

    summary_lines = [
        "KEY FINDINGS FROM ABLATION",
        "",
        "Q1 (Routing before vs after finetuning):",
    ]

    # compute if finetuned shows lower CV (more specialized) than pretrained
    pre_cv = np.nanmean([model_cvs[0], model_cvs[1]]) if len(model_cvs) >= 2 else float('nan')
    ft_cv = np.nanmean([model_cvs[2], model_cvs[3]]) if len(model_cvs) >= 4 else float('nan')
    if not np.isnan(pre_cv) and not np.isnan(ft_cv):
        if ft_cv < pre_cv:
            summary_lines.append(f"  → Finetuning REDUCES load imbalance CV: {pre_cv:.3f} → {ft_cv:.3f}")
            summary_lines.append("    (more balanced routing after finetuning)")
        else:
            summary_lines.append(f"  → Finetuning changes CV: {pre_cv:.3f} → {ft_cv:.3f}")

    summary_lines += [
        "",
        "Q2 (Does single category → expert specialization?):",
        "  → Compare single vs mixed CV in plots 4-7 above.",
        "  → Lower CV in single category = concept-driven specialization.",
        "",
        "E1-E4 scaling:",
        "  → 1 expert = degenerate baseline (no routing possible)",
        "  → 4 experts = best load balance in most configs",
    ]

    ax_text.text(0.05, 0.95, '\n'.join(summary_lines),
                 transform=ax_text.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[✓] Comparison figure saved: {output_path}")
    print(f"    Open with: eog {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="ablation_results",
                        help="Directory containing ablation run subdirs")
    parser.add_argument("--output", type=str, default="ablation_comparison.png",
                        help="Output figure path")
    args = parser.parse_args()
    make_comparison_figure(args.results_dir, args.output)
