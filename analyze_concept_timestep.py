"""
analyze_concept_timestep.py
===========================
Path C experiment: Per-class, per-timestep expert routing analysis.

This answers the critical question:
  "Does the temporal routing pattern change depending on the concept being generated?"

If dogs and geyser show DIFFERENT expert preferences at early timesteps,
concept-temporal specialization exists (even if it's hidden when averaged).

Output:
  1. class_timestep_expert_heatmap.png — the main result
  2. per_class_routing_curves.png — expert utilization over time, one panel per class
  3. concept_routing_stats.json — raw data for further analysis

Usage:
    # Analyze the trained 4-expert model
    CUDA_VISIBLE_DEVICES=1 python analyze_concept_timestep.py \\
        --ckpt results-finetune/004-DiT-XL-2-MoE-finetune-moe24-27/checkpoints/0115000.pt \\
        --num-experts 4 --moe-blocks 24 25 26 27 \\
        --output-dir concept_analysis/finetuned_4exp

    # Analyze pretrained (random gate)
    CUDA_VISIBLE_DEVICES=1 python analyze_concept_timestep.py \\
        --num-experts 4 --moe-blocks 24 25 26 27 \\
        --output-dir concept_analysis/pretrained_4exp

    # Analyze the trained 3-expert model
    CUDA_VISIBLE_DEVICES=1 python analyze_concept_timestep.py \\
        --ckpt results-finetune/004-DiT-XL-2-MoE-finetune-moe24-27/checkpoints/final_3experts_80k.pt \\
        --num-experts 3 --moe-blocks 24 25 26 27 \\
        --output-dir concept_analysis/finetuned_3exp
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models, load_pretrained_with_moe
import argparse
import json
import os


# ImageNet class index -> human-readable name
CLASS_NAMES = {
    207: "Golden Retriever",
    360: "Otter",
    387: "Red Panda",
    974: "Geyser",
    88:  "Macaw",
    979: "Valley",
    417: "Balloon",
    279: "Arctic Fox",
    281: "Tabby Cat",
    388: "Giant Panda",
    340: "Zebra",
    330: "Koala",
    386: "African Elephant",
}


class ConceptRouterHook:
    """
    Enhanced routing hook that preserves the batch dimension so we can
    attribute routing decisions to specific class labels.

    Key difference from RouterHook in analyze_routing.py:
    - Stores topk_idx reshaped as (batch_size, seq_len, top_k)
      instead of flattened (batch*seq_len, top_k)
    - This lets us separate routing by class label after sampling
    """

    def __init__(self, block_idx, gate_module, num_experts):
        self.block_idx = block_idx
        self.gate = gate_module
        self.num_experts = num_experts
        self.original_forward = gate_module.forward
        self.routing_data = []
        gate_module.forward = self._hooked_forward

    def _hooked_forward(self, hidden_states):
        topk_idx, topk_weight, aux_loss = self.original_forward(hidden_states)

        bsz, seq_len, h = hidden_states.shape

        with torch.no_grad():
            flat_h = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = torch.nn.functional.linear(flat_h, self.gate.weight, None)
            probs = logits.softmax(dim=-1)

        self.routing_data.append({
            # Reshape to preserve batch dim: (bsz, seq_len*top_k)
            'topk_idx': topk_idx.detach().cpu().reshape(bsz, seq_len, -1),
            'probs': probs.detach().cpu().reshape(bsz, seq_len, -1),
            'batch_size': bsz,
            'seq_len': seq_len,
        })

        return topk_idx, topk_weight, aux_loss

    def restore(self):
        self.gate.forward = self.original_forward


def install_hooks(model, moe_block_indices):
    hooks = []
    for idx in moe_block_indices:
        block = model.blocks[idx]
        if hasattr(block, 'moe') and hasattr(block.moe, 'gate'):
            num_experts = block.moe.gate.n_routed_experts
            hook = ConceptRouterHook(idx, block.moe.gate, num_experts)
            hooks.append(hook)
    return hooks


def compute_per_class_per_timestep(hooks, class_labels, num_experts, num_timesteps):
    """
    For each hook (MoE block), compute expert utilization broken down by:
      - class label (which image)
      - timestep (denoising step index)

    Returns:
      result[block_idx] = np.array of shape (n_classes, n_timesteps, n_experts)
    """
    n_classes = len(class_labels)
    result = {}

    for hook in hooks:
        # Each routing_data entry = one forward call = one timestep
        # With CFG, batch = [n_classes conditional, n_classes unconditional]
        # We only care about the conditional half (first n_classes images)
        n_steps = len(hook.routing_data)
        data = np.zeros((n_classes, n_steps, num_experts))

        for t, rd in enumerate(hook.routing_data):
            topk_idx = rd['topk_idx'].numpy()  # (bsz, seq_len, top_k)
            bsz = topk_idx.shape[0]

            # Only take conditional half (first n_classes images)
            cond_topk = topk_idx[:n_classes]  # (n_classes, seq_len, top_k)

            for cls_i in range(n_classes):
                # All expert choices for this class at this timestep
                choices = cond_topk[cls_i].flatten()
                for e in range(num_experts):
                    data[cls_i, t, e] = (choices == e).sum()
                # Normalize to fractions
                total = len(choices)
                if total > 0:
                    data[cls_i, t, :] /= total

        result[hook.block_idx] = data

    return result


def plot_class_timestep_heatmap(data, class_labels, num_experts, save_path, title_suffix=""):
    """
    THE MAIN PLOT: For each MoE block, show a heatmap where:
      - Y-axis = (class, expert) pairs
      - X-axis = timestep
      - Color = utilization fraction

    This reveals whether different classes route to different experts at different timesteps.
    """
    blocks = sorted(data.keys())
    n_classes = len(class_labels)
    n_blocks = len(blocks)

    fig, axes = plt.subplots(n_blocks, 1, figsize=(16, 4 * n_blocks + 2),
                              squeeze=False)
    fig.suptitle(f"Per-Class Expert Routing Over Denoising Time{title_suffix}",
                 fontsize=14, fontweight='bold', y=1.01)

    for ax_i, block_idx in enumerate(blocks):
        ax = axes[ax_i, 0]
        block_data = data[block_idx]  # (n_classes, n_timesteps, n_experts)
        n_steps = block_data.shape[1]

        # Stack into (n_classes * n_experts, n_timesteps) for heatmap
        heatmap = np.zeros((n_classes * num_experts, n_steps))
        y_labels = []
        for cls_i in range(n_classes):
            cls_name = CLASS_NAMES.get(class_labels[cls_i], f"Class {class_labels[cls_i]}")
            short_name = cls_name[:12]
            for e in range(num_experts):
                row = cls_i * num_experts + e
                heatmap[row] = block_data[cls_i, :, e]
                y_labels.append(f"{short_name} → E{e}")

        im = ax.imshow(heatmap, aspect='auto', cmap='YlOrRd', vmin=0,
                       interpolation='nearest')

        # Y-axis
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.set_ylabel('Class → Expert')

        # X-axis
        step_ticks = np.linspace(0, n_steps - 1, min(10, n_steps)).astype(int)
        ax.set_xticks(step_ticks)
        ax.set_xticklabels([f"t={s}" for s in step_ticks], fontsize=8)
        ax.set_xlabel('Denoising Timestep (early → late)')

        ax.set_title(f"Block {block_idx}", fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label='Utilization')

        # Add horizontal lines between classes
        for cls_i in range(1, n_classes):
            ax.axhline(y=cls_i * num_experts - 0.5, color='white', linewidth=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_class_routing_curves(data, class_labels, num_experts, save_path,
                                   block_idx_to_show=None, title_suffix=""):
    """
    For one block (default: last), show expert utilization curves over time,
    one subplot per class. This is the most readable way to compare.
    """
    blocks = sorted(data.keys())
    if block_idx_to_show is None:
        block_idx_to_show = blocks[-1]  # last MoE block

    block_data = data[block_idx_to_show]  # (n_classes, n_timesteps, n_experts)
    n_classes = len(class_labels)
    n_steps = block_data.shape[1]

    cols = 4
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    fig.suptitle(f"Block {block_idx_to_show}: Expert Routing per Class{title_suffix}",
                 fontsize=13, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, num_experts))
    timesteps = np.arange(n_steps)

    for cls_i in range(n_classes):
        r, c = cls_i // cols, cls_i % cols
        ax = axes[r, c]
        cls_name = CLASS_NAMES.get(class_labels[cls_i], f"Class {class_labels[cls_i]}")

        for e in range(num_experts):
            curve = block_data[cls_i, :, e]
            # Smooth with moving average for readability
            if len(curve) > 5:
                kernel = np.ones(5) / 5
                curve_smooth = np.convolve(curve, kernel, mode='same')
            else:
                curve_smooth = curve
            ax.plot(timesteps, curve_smooth, color=colors[e],
                    label=f'E{e}', linewidth=1.5, alpha=0.85)

        ax.set_title(cls_name, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 0.8 if num_experts <= 2 else 0.6)
        ax.set_xlabel('Step', fontsize=8)
        ax.set_ylabel('Utilization', fontsize=8)
        ax.legend(fontsize=7, ncol=num_experts, loc='upper right')
        ax.grid(alpha=0.2)

        # Reference line for uniform routing
        uniform = 1.0 / num_experts
        ax.axhline(y=uniform, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    # Hide empty subplots
    for cls_i in range(n_classes, rows * cols):
        r, c = cls_i // cols, cls_i % cols
        axes[r, c].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def compute_concept_divergence(data, class_labels, num_experts):
    """
    Compute how DIFFERENT the routing patterns are across classes.

    For each block and timestep, compute the Jensen-Shannon divergence
    between each pair of classes' expert distributions.

    High divergence = classes route to different experts = concept specialization!
    Low divergence = all classes route identically = no concept specialization.
    """
    from scipy.spatial.distance import jensenshannon

    blocks = sorted(data.keys())
    n_classes = len(class_labels)
    result = {}

    for block_idx in blocks:
        block_data = data[block_idx]  # (n_classes, n_timesteps, n_experts)
        n_steps = block_data.shape[1]

        # Pairwise JSD at each timestep
        jsd_over_time = np.zeros(n_steps)
        n_pairs = 0
        for t in range(n_steps):
            jsd_sum = 0.0
            pair_count = 0
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    p = block_data[i, t, :] + 1e-10
                    q = block_data[j, t, :] + 1e-10
                    p = p / p.sum()
                    q = q / q.sum()
                    jsd_sum += jensenshannon(p, q) ** 2  # squared JSD
                    pair_count += 1
            jsd_over_time[t] = jsd_sum / max(pair_count, 1)

        result[block_idx] = jsd_over_time

    return result


def plot_divergence(jsd_data, save_path, title_suffix=""):
    """Plot the concept divergence (JSD) over denoising timesteps for each block."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(jsd_data)))

    for i, (block_idx, jsd_curve) in enumerate(sorted(jsd_data.items())):
        timesteps = np.arange(len(jsd_curve))
        # Smooth
        if len(jsd_curve) > 5:
            kernel = np.ones(5) / 5
            smooth = np.convolve(jsd_curve, kernel, mode='same')
        else:
            smooth = jsd_curve
        ax.plot(timesteps, smooth, label=f'Block {block_idx}',
                color=colors[i], linewidth=2)

    ax.set_xlabel('Denoising Timestep (early → late)', fontsize=11)
    ax.set_ylabel('Avg Pairwise JSD Between Classes\n(higher = more concept-specific routing)',
                  fontsize=10)
    ax.set_title(f'Concept Routing Divergence Over Time{title_suffix}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_size = args.image_size // 8
    moe_blocks = args.moe_blocks

    print(f"\n{'='*60}")
    print(f"  CONCEPT × TIMESTEP ROUTING ANALYSIS")
    print(f"{'='*60}")
    print(f"  MoE blocks: {moe_blocks}")
    print(f"  Num experts: {args.num_experts}")
    print(f"  Sampling steps: {args.num_sampling_steps}")

    # Build model
    model = DiT_models['DiT-XL/2-MoE'](
        input_size=latent_size,
        num_classes=args.num_classes,
        moe_blocks=moe_blocks,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        n_shared_experts=args.n_shared_experts,
        rank=args.rank,
        use_dwconv=not args.no_dwconv,
    ).to(device)

    # Load checkpoint
    ckpt_path = args.ckpt or f"pretrained_models/DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    print(f"  Checkpoint: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "ema_trainable_only" in raw:
        print(f"  → Finetuned (EMA) at step={raw.get('train_steps', '?')}")
        backbone_state = find_model("pretrained_models/DiT-XL-2-256x256.pt")
        load_pretrained_with_moe(model, backbone_state)
        model.load_state_dict(raw["ema_trainable_only"], strict=False)
    elif isinstance(raw, dict) and "model_trainable_only" in raw:
        print(f"  → Finetuned (model) at step={raw.get('train_steps', '?')}")
        backbone_state = find_model("pretrained_models/DiT-XL-2-256x256.pt")
        load_pretrained_with_moe(model, backbone_state)
        model.load_state_dict(raw["model_trainable_only"], strict=False)
    else:
        print(f"  → Pretrained (vanilla DiT)")
        load_pretrained_with_moe(model, raw)
    model.eval()

    # Install hooks
    hooks = install_hooks(model, moe_blocks)
    print(f"  Hooks installed on {len(hooks)} blocks")

    # Diffusion + VAE
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Sampling
    class_labels = args.class_labels
    print(f"  Class labels: {class_labels}")
    class_names = [CLASS_NAMES.get(c, f"Class {c}") for c in class_labels]
    print(f"  Classes: {class_names}")

    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # CFG
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    print(f"\nSampling ({args.num_sampling_steps} steps)...")
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=True, device=device
    )

    # Save sample image
    samples_img, _ = samples.chunk(2, dim=0)
    samples_img = vae.decode(samples_img / 0.18215).sample
    from torchvision.utils import save_image
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    save_image(samples_img, os.path.join(out_dir, "samples.png"),
               nrow=4, normalize=True, value_range=(-1, 1))

    # ── ANALYSIS ──────────────────────────────────────────────────────────────
    print(f"\nAnalyzing per-class per-timestep routing...")
    num_experts = args.num_experts

    # 1. Compute per-class per-timestep expert utilization
    per_class_data = compute_per_class_per_timestep(
        hooks, class_labels, num_experts, args.num_sampling_steps)

    # 2. Main heatmap
    plot_class_timestep_heatmap(
        per_class_data, class_labels, num_experts,
        os.path.join(out_dir, "class_timestep_expert_heatmap.png"))

    # 3. Per-class routing curves (one panel per block)
    for block_idx in sorted(per_class_data.keys()):
        plot_per_class_routing_curves(
            per_class_data, class_labels, num_experts,
            os.path.join(out_dir, f"per_class_routing_block{block_idx}.png"),
            block_idx_to_show=block_idx)

    # 4. Concept divergence (JSD)
    try:
        jsd_data = compute_concept_divergence(
            per_class_data, class_labels, num_experts)
        plot_divergence(jsd_data, os.path.join(out_dir, "concept_divergence_jsd.png"))
    except ImportError:
        print("  [!] scipy not installed — skipping JSD divergence plot")
        print("      Install with: pip install scipy")
        jsd_data = {}

    # 5. Save raw stats
    stats = {
        "class_labels": class_labels,
        "class_names": class_names,
        "num_experts": num_experts,
        "num_timesteps": args.num_sampling_steps,
    }
    # Per-class per-block summary: average expert utilization across all timesteps
    for block_idx, block_data in per_class_data.items():
        stats[f"block_{block_idx}"] = {}
        for cls_i, cls_label in enumerate(class_labels):
            cls_name = CLASS_NAMES.get(cls_label, str(cls_label))
            avg_util = block_data[cls_i].mean(axis=0).tolist()
            stats[f"block_{block_idx}"][cls_name] = {
                f"E{e}": f"{avg_util[e]:.3f}" for e in range(num_experts)
            }

    # Per-class dominant expert
    stats["dominant_experts"] = {}
    for cls_i, cls_label in enumerate(class_labels):
        cls_name = CLASS_NAMES.get(cls_label, str(cls_label))
        dominance = {}
        for block_idx, block_data in per_class_data.items():
            # Which expert has highest utilization for this class, averaged over time?
            avg = block_data[cls_i].mean(axis=0)
            dominance[f"block_{block_idx}"] = {
                "dominant": int(np.argmax(avg)),
                "fraction": f"{avg.max():.3f}",
            }
        stats["dominant_experts"][cls_name] = dominance

    stats_path = os.path.join(out_dir, "concept_routing_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    # ── PRINT SUMMARY ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CONCEPT ROUTING SUMMARY")
    print(f"{'='*60}")

    for block_idx in sorted(per_class_data.keys()):
        print(f"\n  Block {block_idx}:")
        block_data = per_class_data[block_idx]
        for cls_i, cls_label in enumerate(class_labels):
            cls_name = CLASS_NAMES.get(cls_label, str(cls_label))
            avg_util = block_data[cls_i].mean(axis=0)
            dominant = np.argmax(avg_util)
            util_str = ", ".join(f"E{e}={avg_util[e]:.1%}" for e in range(num_experts))
            print(f"    {cls_name:<20s}: {util_str}  (dominant: E{dominant})")

    # Check: do different classes prefer different experts?
    print(f"\n  Concept Specialization Check:")
    for block_idx in sorted(per_class_data.keys()):
        block_data = per_class_data[block_idx]
        dominant_experts = []
        for cls_i in range(len(class_labels)):
            avg = block_data[cls_i].mean(axis=0)
            dominant_experts.append(np.argmax(avg))
        unique = len(set(dominant_experts))
        total = len(dominant_experts)
        print(f"    Block {block_idx}: {unique}/{total} unique dominant experts "
              f"{'← SPECIALIZATION DETECTED!' if unique > 1 else '← uniform (no specialization)'}")

    if jsd_data:
        print(f"\n  Concept Divergence (avg JSD):")
        for block_idx, jsd_curve in sorted(jsd_data.items()):
            early_jsd = jsd_curve[:len(jsd_curve)//4].mean()
            late_jsd = jsd_curve[-len(jsd_curve)//4:].mean()
            print(f"    Block {block_idx}: early={early_jsd:.4f}, late={late_jsd:.4f} "
                  f"{'← early > late!' if early_jsd > late_jsd * 1.2 else ''}")

    # Cleanup
    for hook in hooks:
        hook.restore()

    print(f"\n{'='*60}")
    print(f"  All results saved to: {out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="concept_analysis")

    parser.add_argument("--moe-blocks", type=int, nargs="+", default=[24, 25, 26, 27])
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--n-shared-experts", type=int, default=2)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--no-dwconv", action="store_true")

    parser.add_argument("--class-labels", type=int, nargs="+",
                        default=[207, 360, 387, 974, 88, 979, 417, 279],
                        help="ImageNet class labels to analyze")

    args = parser.parse_args()
    main(args)
