"""
analyze_cdmoe_result.py
=======================
Loads the base Diff-MoE model, overlays CD-MoE trained gate weights,
then runs the same concept-timestep analysis.

Since mode=loss_only, the gate architecture is identical to standard MoEGate
(just the weight matrix). Only the loss function differed during training.
So we can load the trained gate weights directly into the standard model.

Usage:
    python analyze_cdmoe_result.py \
        --base-ckpt results-finetune/004-DiT-XL-2-MoE-finetune-moe24-27/checkpoints/0115000.pt \
        --cdmoe-ckpt results-cdmoe/001-CDMoE-gate-loss_only/checkpoints/final.pt \
        --output-dir concept_analysis/cdmoe_lossonly
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import os
from models import DiT_models, load_pretrained_with_moe
from download import find_model

# Reuse everything from analyze_concept_timestep
from analyze_concept_timestep import (
    install_hooks, compute_per_class_per_timestep,
    plot_class_timestep_heatmap, plot_per_class_routing_curves,
    compute_concept_divergence, plot_divergence,
    CLASS_NAMES,
)
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
import numpy as np
import json


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_size = args.image_size // 8
    moe_blocks = args.moe_blocks

    print(f"\n{'='*60}")
    print(f"  CD-MoE CONCEPT × TIMESTEP ANALYSIS")
    print(f"{'='*60}")

    # ── Step 1: Build model and load base checkpoint ─────────────────────
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

    print(f"  Loading base model: {args.base_ckpt}")
    raw = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "ema_trainable_only" in raw:
        print(f"  → Finetuned (EMA) at step={raw.get('train_steps', '?')}")
        backbone_state = find_model("pretrained_models/DiT-XL-2-256x256.pt")
        load_pretrained_with_moe(model, backbone_state)
        model.load_state_dict(raw["ema_trainable_only"], strict=False)
    elif isinstance(raw, dict) and "model_trainable_only" in raw:
        backbone_state = find_model("pretrained_models/DiT-XL-2-256x256.pt")
        load_pretrained_with_moe(model, backbone_state)
        model.load_state_dict(raw["model_trainable_only"], strict=False)
    else:
        load_pretrained_with_moe(model, raw)

    # ── Step 2: Overlay CD-MoE trained gate weights ──────────────────────
    print(f"  Loading CD-MoE gate weights: {args.cdmoe_ckpt}")
    cdmoe = torch.load(args.cdmoe_ckpt, map_location="cpu", weights_only=False)
    print(f"  → Mode: {cdmoe.get('mode')}, Steps: {cdmoe.get('train_steps')}")

    gates_loaded = 0
    for key, state in cdmoe['gate_state'].items():
        # key is like 'blocks.24.moe.gate'
        block_idx = int(key.split('.')[1])
        gate = model.blocks[block_idx].moe.gate
        # Load the trained routing weight
        gate.weight.data.copy_(state['weight'].to(device))
        gates_loaded += 1
        print(f"  → Loaded gate weights for block {block_idx}")

    print(f"  Loaded {gates_loaded} CD-MoE gates")
    model.eval()

    # ── Step 3: Install hooks and run sampling ───────────────────────────
    hooks = install_hooks(model, moe_blocks)
    print(f"  Hooks installed on {len(hooks)} blocks")

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    class_labels = args.class_labels
    print(f"  Class labels: {class_labels}")
    class_names = [CLASS_NAMES.get(c, f"Class {c}") for c in class_labels]

    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    print(f"\nSampling ({args.num_sampling_steps} steps)...")
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=True, device=device
    )

    # Save samples
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    samples_img, _ = samples.chunk(2, dim=0)
    samples_img = vae.decode(samples_img / 0.18215).sample
    save_image(samples_img, os.path.join(out_dir, "samples.png"),
               nrow=4, normalize=True, value_range=(-1, 1))

    # ── Step 4: Analysis ─────────────────────────────────────────────────
    print(f"\nAnalyzing per-class per-timestep routing...")
    num_experts = args.num_experts

    per_class_data = compute_per_class_per_timestep(
        hooks, class_labels, num_experts, args.num_sampling_steps)

    plot_class_timestep_heatmap(
        per_class_data, class_labels, num_experts,
        os.path.join(out_dir, "class_timestep_expert_heatmap.png"))

    for block_idx in sorted(per_class_data.keys()):
        plot_per_class_routing_curves(
            per_class_data, class_labels, num_experts,
            os.path.join(out_dir, f"per_class_routing_block{block_idx}.png"),
            block_idx_to_show=block_idx)

    try:
        jsd_data = compute_concept_divergence(
            per_class_data, class_labels, num_experts)
        plot_divergence(jsd_data, os.path.join(out_dir, "concept_divergence_jsd.png"))
    except ImportError:
        jsd_data = {}

    # Save stats
    stats = {
        "class_labels": class_labels,
        "class_names": class_names,
        "num_experts": num_experts,
        "cdmoe_mode": cdmoe.get('mode'),
        "cdmoe_steps": cdmoe.get('train_steps'),
        "base_ckpt": args.base_ckpt,
    }

    for block_idx, block_data in per_class_data.items():
        stats[f"block_{block_idx}"] = {}
        for cls_i, cls_label in enumerate(class_labels):
            cls_name = CLASS_NAMES.get(cls_label, str(cls_label))
            avg_util = block_data[cls_i].mean(axis=0).tolist()
            stats[f"block_{block_idx}"][cls_name] = {
                f"E{e}": f"{avg_util[e]:.3f}" for e in range(num_experts)
            }

    stats["dominant_experts"] = {}
    for cls_i, cls_label in enumerate(class_labels):
        cls_name = CLASS_NAMES.get(cls_label, str(cls_label))
        dominance = {}
        for block_idx, block_data in per_class_data.items():
            avg = block_data[cls_i].mean(axis=0)
            dominance[f"block_{block_idx}"] = {
                "dominant": int(np.argmax(avg)),
                "fraction": f"{avg.max():.3f}",
            }
        stats["dominant_experts"][cls_name] = dominance

    with open(os.path.join(out_dir, "concept_routing_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CD-MoE CONCEPT ROUTING SUMMARY")
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
            print(f"    Block {block_idx}: early={early_jsd:.4f}, late={late_jsd:.4f}")

    # Compare with baseline
    baseline_path = "concept_analysis/finetuned_4exp/concept_routing_stats.json"
    if os.path.exists(baseline_path):
        print(f"\n  ── COMPARISON vs Standard Diff-MoE ──")
        with open(baseline_path) as f:
            baseline = json.load(f)
        for block_key in [k for k in stats if k.startswith("block_")]:
            block_idx = block_key.split("_")[1]
            if block_key in baseline:
                print(f"\n    Block {block_idx}:")
                for cls_name in stats[block_key]:
                    if cls_name in baseline[block_key]:
                        cd_vals = stats[block_key][cls_name]
                        bl_vals = baseline[block_key][cls_name]
                        cd_dom = max(cd_vals.values())
                        bl_dom = max(bl_vals.values())
                        arrow = "↑" if float(cd_dom) > float(bl_dom) else "↓"
                        print(f"      {cls_name:<18s}: baseline max={bl_dom} → CD-MoE max={cd_dom} {arrow}")

    for hook in hooks:
        hook.restore()

    print(f"\n{'='*60}")
    print(f"  All results saved to: {out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", type=str, required=True,
                        help="Path to Stage 1 finetuned checkpoint (full MoE model)")
    parser.add_argument("--cdmoe-ckpt", type=str, required=True,
                        help="Path to CD-MoE gate checkpoint")
    parser.add_argument("--output-dir", type=str, default="concept_analysis/cdmoe_lossonly")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae", type=str, default="mse")
    parser.add_argument("--moe-blocks", type=int, nargs="+", default=[24, 25, 26, 27])
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--n-shared-experts", type=int, default=2)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--no-dwconv", action="store_true")
    parser.add_argument("--class-labels", type=int, nargs="+",
                        default=[207, 360, 387, 974, 88, 979, 417, 279])

    args = parser.parse_args()
    main(args)
