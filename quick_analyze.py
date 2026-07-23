"""
quick_analyze.py — Fast routing analysis for grid search.

Loads base model + CD-MoE gates, runs routing analysis on 8 test classes,
outputs metrics as JSON. No image generation — purely routing metrics.

Usage:
    python quick_analyze.py \
        --base-ckpt results-finetune/004-.../0115000.pt \
        --cdmoe-ckpt results-cdmoe/<run>/checkpoints/final.pt \
        --output metrics.json
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import json
import os
import numpy as np
from diffusion import create_diffusion
from download import find_model
from models import DiT_models, load_pretrained_with_moe


# 8 diverse test classes
TEST_CLASSES = [207, 360, 387, 974, 88, 979, 417, 279]
CLASS_NAMES = {
    207: "Golden_Retriever", 360: "Otter", 387: "Red_Panda",
    974: "Geyser", 88: "Macaw", 979: "Valley", 417: "Balloon", 279: "Arctic_Fox",
}


def load_model_with_gates(args, device):
    """Load base model and overlay CD-MoE gate weights."""
    latent_size = args.image_size // 8
    moe_blocks = args.moe_blocks

    model = DiT_models['DiT-XL/2-MoE'](
        input_size=latent_size, num_classes=1000,
        moe_blocks=moe_blocks, num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        n_shared_experts=args.n_shared_experts,
        rank=args.rank, use_dwconv=not args.no_dwconv,
    ).to(device)

    # Load base checkpoint
    raw = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "ema_trainable_only" in raw:
        backbone_state = find_model("pretrained_models/DiT-XL-2-256x256.pt")
        load_pretrained_with_moe(model, backbone_state)
        model.load_state_dict(raw["ema_trainable_only"], strict=False)
    elif isinstance(raw, dict) and "model_trainable_only" in raw:
        backbone_state = find_model("pretrained_models/DiT-XL-2-256x256.pt")
        load_pretrained_with_moe(model, backbone_state)
        model.load_state_dict(raw["model_trainable_only"], strict=False)
    else:
        load_pretrained_with_moe(model, raw)

    # Overlay CD-MoE gate weights
    if args.cdmoe_ckpt:
        cdmoe = torch.load(args.cdmoe_ckpt, map_location="cpu", weights_only=False)
        for key, state in cdmoe['gate_state'].items():
            block_idx = int(key.split('.')[1])
            gate = model.blocks[block_idx].moe.gate
            gate.weight.data.copy_(state['weight'].to(device))

    model.eval()
    return model


def analyze_routing(model, diffusion, moe_blocks, device, num_timesteps=50):
    """Run routing analysis across test classes and timesteps."""
    latent_size = 32  # 256 // 8
    n_experts = model.blocks[moe_blocks[0]].moe.gate.n_routed_experts

    # Storage: [block][class][timestep] -> expert distribution
    routing_data = {}
    for block_idx in moe_blocks:
        routing_data[block_idx] = {}

    # Hook to capture routing decisions
    routing_captures = {}

    def make_hook(block_idx):
        def hook_fn(module, input, output):
            topk_idx, topk_weight, aux_loss = output
            routing_captures[block_idx] = topk_idx.detach()
        return hook_fn

    # Register hooks
    hooks = []
    for idx in moe_blocks:
        h = model.blocks[idx].moe.gate.register_forward_hook(make_hook(idx))
        hooks.append(h)

    timesteps = torch.linspace(0, 999, num_timesteps, dtype=torch.long, device=device)

    with torch.no_grad():
        for cls_label in TEST_CLASSES:
            cls_name = CLASS_NAMES[cls_label]

            for block_idx in moe_blocks:
                routing_data[block_idx][cls_name] = []

            # Generate samples with CFG
            n_samples = 4
            z = torch.randn(n_samples, 4, latent_size, latent_size, device=device)
            y = torch.tensor([cls_label] * n_samples, device=device)

            # CFG setup
            z_cfg = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n_samples, device=device)
            y_cfg = torch.cat([y, y_null], 0)

            for t_val in timesteps:
                t = t_val.expand(n_samples * 2)
                noise = torch.randn_like(z_cfg)

                # Forward pass
                routing_captures.clear()
                _ = model(z_cfg, t, y_cfg)

                # Collect routing for real (non-null) samples
                for block_idx in moe_blocks:
                    if block_idx in routing_captures:
                        topk = routing_captures[block_idx][:n_samples]  # real only
                        # Expert utilization
                        flat = topk.reshape(-1)
                        counts = torch.zeros(n_experts, device=device)
                        for e in range(n_experts):
                            counts[e] = (flat == e).float().sum()
                        dist = counts / counts.sum()
                        routing_data[block_idx][cls_name].append(dist.cpu().numpy())

    # Remove hooks
    for h in hooks:
        h.remove()

    return routing_data


def compute_metrics(routing_data, moe_blocks, n_experts):
    """Compute all grid search evaluation metrics."""
    metrics = {"blocks": {}}

    all_unique = []
    all_max_util = []
    all_min_util = []
    all_jsd_early = []
    all_jsd_ratio = []

    for block_idx in moe_blocks:
        block_data = routing_data[block_idx]
        classes = list(block_data.keys())

        # Per-class average routing distribution
        class_avg_dists = {}
        for cls_name in classes:
            dists = np.array(block_data[cls_name])  # (T, E)
            class_avg_dists[cls_name] = dists.mean(axis=0)

        # 1. Unique dominant experts
        dominants = {}
        for cls_name, avg_dist in class_avg_dists.items():
            dom_expert = int(np.argmax(avg_dist))
            dominants[cls_name] = dom_expert
        unique_dom = len(set(dominants.values()))

        # 2. Max utilization per class
        max_utils = {cls: float(np.max(d)) for cls, d in class_avg_dists.items()}
        avg_max_util = np.mean(list(max_utils.values()))

        # 3. Min expert utilization (global)
        global_dist = np.mean([d for d in class_avg_dists.values()], axis=0)
        min_util = float(np.min(global_dist))

        # 4. JSD between classes at early vs late timesteps
        n_timesteps = len(block_data[classes[0]])
        early_range = range(0, min(10, n_timesteps))
        late_range = range(max(0, n_timesteps - 10), n_timesteps)

        def avg_pairwise_jsd(time_range):
            jsd_sum = 0
            n_pairs = 0
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    dists_i = np.array(block_data[classes[i]])
                    dists_j = np.array(block_data[classes[j]])
                    for t in time_range:
                        if t < len(dists_i) and t < len(dists_j):
                            p = dists_i[t] + 1e-10
                            q = dists_j[t] + 1e-10
                            m = 0.5 * (p + q)
                            jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
                            jsd_sum += jsd
                            n_pairs += 1
            return jsd_sum / max(n_pairs, 1)

        jsd_early = avg_pairwise_jsd(early_range)
        jsd_late = avg_pairwise_jsd(late_range)
        jsd_ratio = jsd_early / max(jsd_late, 1e-10)

        # 5. Mean pairwise KL between class average distributions
        kl_sum = 0
        kl_pairs = 0
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                p = class_avg_dists[classes[i]] + 1e-10
                q = class_avg_dists[classes[j]] + 1e-10
                kl_pq = np.sum(p * np.log(p / q))
                kl_qp = np.sum(q * np.log(q / p))
                kl_sum += 0.5 * (kl_pq + kl_qp)
                kl_pairs += 1
        mean_kl = kl_sum / max(kl_pairs, 1)

        block_metrics = {
            "unique_dominant": unique_dom,
            "dominants": dominants,
            "avg_max_util": round(avg_max_util, 4),
            "min_expert_util": round(min_util, 4),
            "jsd_early": round(jsd_early, 6),
            "jsd_late": round(jsd_late, 6),
            "jsd_ratio": round(jsd_ratio, 2),
            "mean_pairwise_kl": round(mean_kl, 6),
            "max_utils_per_class": {k: round(v, 4) for k, v in max_utils.items()},
        }
        metrics["blocks"][str(block_idx)] = block_metrics

        all_unique.append(unique_dom)
        all_max_util.append(avg_max_util)
        all_min_util.append(min_util)
        all_jsd_early.append(jsd_early)
        all_jsd_ratio.append(jsd_ratio)

    # Summary metrics
    metrics["summary"] = {
        "avg_unique_dominant": round(np.mean(all_unique), 2),
        "total_unique_dominant": sum(all_unique),
        "avg_max_util": round(np.mean(all_max_util), 4),
        "worst_min_util": round(min(all_min_util), 4),
        "avg_jsd_early": round(np.mean(all_jsd_early), 6),
        "avg_jsd_ratio": round(np.mean(all_jsd_ratio), 2),
        # Composite score: weighted combination (higher = better)
        "composite_score": round(
            sum(all_unique) * 2.0  # reward unique experts
            + np.mean(all_max_util) * 10.0  # reward strong preferences
            + np.mean(all_jsd_early) * 1000.0  # reward concept divergence
            - max(0, 0.125 - min(all_min_util)) * 100.0  # penalize collapse
            , 4),
    }

    return metrics


def main(args):
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    moe_blocks = args.moe_blocks
    n_experts = args.num_experts

    print(f"Loading model...")
    model = load_model_with_gates(args, device)

    print(f"Creating diffusion...")
    diffusion = create_diffusion("250")

    print(f"Analyzing routing across {len(TEST_CLASSES)} classes, {args.num_timesteps} timesteps...")
    routing_data = analyze_routing(model, diffusion, moe_blocks, device, args.num_timesteps)

    print(f"Computing metrics...")
    metrics = compute_metrics(routing_data, moe_blocks, n_experts)

    # Add config info
    metrics["config"] = {
        "base_ckpt": args.base_ckpt,
        "cdmoe_ckpt": args.cdmoe_ckpt,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  ROUTING METRICS SUMMARY")
    print(f"{'='*60}")
    for block_idx in moe_blocks:
        bm = metrics["blocks"][str(block_idx)]
        print(f"\n  Block {block_idx}:")
        print(f"    Unique dominant: {bm['unique_dominant']}/8")
        print(f"    Avg max util:    {bm['avg_max_util']:.1%}")
        print(f"    Min expert util: {bm['min_expert_util']:.1%}")
        print(f"    JSD early:       {bm['jsd_early']:.6f}")
        print(f"    JSD ratio:       {bm['jsd_ratio']:.1f}×")
        print(f"    Mean KL:         {bm['mean_pairwise_kl']:.6f}")

    s = metrics["summary"]
    print(f"\n  {'─'*40}")
    print(f"  OVERALL:")
    print(f"    Total unique dominant: {s['total_unique_dominant']}/32")
    print(f"    Avg max utilization:   {s['avg_max_util']:.1%}")
    print(f"    Worst min util:        {s['worst_min_util']:.1%}")
    print(f"    Avg JSD early:         {s['avg_jsd_early']:.6f}")
    print(f"    Composite score:       {s['composite_score']:.2f}")
    print(f"{'='*60}")

    # Save
    output = args.output or "metrics.json"
    with open(output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", type=str, required=True)
    parser.add_argument("--cdmoe-ckpt", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--moe-blocks", type=int, nargs="+", default=[24, 25, 26, 27])
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--n-shared-experts", type=int, default=2)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--no-dwconv", action="store_true")
    args = parser.parse_args()
    main(args)
