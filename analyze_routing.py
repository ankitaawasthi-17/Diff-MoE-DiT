"""
Analyze MoE expert routing patterns during diffusion sampling.

This script instruments the MoE gate (router) in each DiTBlock_MoE during
a full sampling pass, collecting:
  - Which experts each token is routed to, at every block and every timestep
  - Router probability distributions (before top-k selection)
  - Expert utilization statistics

Then produces visualizations:
  1. Expert utilization heatmap (block × expert) — shows load balance
  2. Expert utilization over time (timestep × expert) — shows temporal specialization
  3. Spatial routing map — which image regions go to which experts
  4. Router entropy over time — measures routing confidence/diversity

Usage:
    conda activate dit
    python analyze_routing.py --num-sampling-steps 50
    python analyze_routing.py --num-sampling-steps 50 --moe-blocks 24 25 26 27
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for SSH
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models, load_pretrained_with_moe
import argparse
import os


class RouterHook:
    """
    Wraps a MoEGate's forward method to capture routing decisions at every call.
    """
    def __init__(self, block_idx, gate_module, num_experts):
        self.block_idx = block_idx
        self.gate = gate_module
        self.num_experts = num_experts
        self.original_forward = gate_module.forward

        # Storage for routing data across all timesteps
        self.routing_data = []  # list of dicts per timestep call

        # Install the hook
        gate_module.forward = self._hooked_forward

    def _hooked_forward(self, hidden_states):
        # Call original forward
        topk_idx, topk_weight, aux_loss = self.original_forward(hidden_states)

        # Record routing decisions (detach to avoid memory leaks)
        bsz, seq_len, h = hidden_states.shape if hidden_states.dim() == 3 else (
            hidden_states.shape[0] // self.gate.top_k, -1, hidden_states.shape[-1]
        )

        # Compute full probability distribution (before top-k)
        with torch.no_grad():
            flat_h = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = torch.nn.functional.linear(flat_h, self.gate.weight, None)
            probs = logits.softmax(dim=-1)  # (batch*seq_len, num_experts)

        self.routing_data.append({
            'topk_idx': topk_idx.detach().cpu(),      # (batch*seq_len, top_k)
            'topk_weight': topk_weight.detach().cpu(), # (batch*seq_len, top_k)
            'probs': probs.detach().cpu(),             # (batch*seq_len, num_experts)
            'batch_size': bsz if isinstance(bsz, int) else bsz,
        })

        return topk_idx, topk_weight, aux_loss

    def restore(self):
        """Remove the hook and restore original forward."""
        self.gate.forward = self.original_forward


def install_hooks(model, moe_block_indices):
    """Install routing hooks on all MoE blocks. Returns list of RouterHook objects."""
    hooks = []
    for idx in moe_block_indices:
        block = model.blocks[idx]
        if hasattr(block, 'moe') and hasattr(block.moe, 'gate'):
            num_experts = block.moe.gate.n_routed_experts
            hook = RouterHook(idx, block.moe.gate, num_experts)
            hooks.append(hook)
    return hooks


def plot_expert_utilization_heatmap(hooks, save_path, num_experts):
    """
    Plot 1: Heatmap of average expert utilization per block.
    Shows whether routing is balanced across experts within each block.
    """
    num_blocks = len(hooks)
    utilization = np.zeros((num_blocks, num_experts))

    for i, hook in enumerate(hooks):
        counts = np.zeros(num_experts)
        total = 0
        for data in hook.routing_data:
            idx = data['topk_idx'].numpy().flatten()
            for e in range(num_experts):
                counts[e] += (idx == e).sum()
            total += len(idx)
        if total > 0:
            utilization[i] = counts / total

    fig, ax = plt.subplots(figsize=(max(8, num_experts), max(4, num_blocks * 0.5 + 1)))
    im = ax.imshow(utilization, aspect='auto', cmap='YlOrRd', vmin=0)
    ax.set_xlabel('Expert Index', fontsize=12)
    ax.set_ylabel('Block Index', fontsize=12)
    ax.set_title('Expert Utilization by Block\n(fraction of tokens routed to each expert)', fontsize=13)
    ax.set_xticks(range(num_experts))
    ax.set_yticks(range(num_blocks))
    ax.set_yticklabels([f'Block {h.block_idx}' for h in hooks])

    # Annotate cells with values
    for i in range(num_blocks):
        for j in range(num_experts):
            val = utilization[i, j]
            color = 'white' if val > 0.15 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label='Fraction of tokens')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_temporal_routing(hooks, save_path, num_experts, num_sampling_steps):
    """
    Plot 2: How expert utilization changes over denoising timesteps.
    Each subplot is one MoE block; x-axis is timestep (t=T→0), y-axis is expert fraction.
    This shows whether different experts specialize for different denoising phases.
    """
    num_blocks = len(hooks)
    fig, axes = plt.subplots(num_blocks, 1, figsize=(12, 3 * num_blocks), sharex=True)
    if num_blocks == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, num_experts))

    for i, (hook, ax) in enumerate(zip(hooks, axes)):
        # Group routing data by timestep
        # The diffusion sampler calls forward once per timestep, but CFG doubles
        # the batch. Each hook.routing_data entry corresponds to one forward call.
        # With CFG, we get 2 calls per timestep (cond + uncond are batched together).
        num_calls = len(hook.routing_data)

        # Compute expert fraction at each call
        expert_fracs = np.zeros((num_calls, num_experts))
        for t, data in enumerate(hook.routing_data):
            idx = data['topk_idx'].numpy().flatten()
            total = len(idx)
            for e in range(num_experts):
                expert_fracs[t, e] = (idx == e).sum() / max(total, 1)

        x = np.arange(num_calls)
        for e in range(num_experts):
            ax.plot(x, expert_fracs[:, e], color=colors[e], alpha=0.8,
                    linewidth=1.5, label=f'Expert {e}')

        ax.set_ylabel('Token fraction', fontsize=10)
        ax.set_title(f'Block {hook.block_idx}', fontsize=11)
        ax.set_ylim(0, max(0.5, expert_fracs.max() * 1.1))
        ax.legend(loc='upper right', fontsize=7, ncol=min(4, num_experts))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Sampling step (T → 0)', fontsize=11)
    fig.suptitle('Expert Routing Over Denoising Timesteps', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_router_entropy(hooks, save_path, num_experts):
    """
    Plot 3: Router entropy over time. High entropy = tokens spread evenly across experts.
    Low entropy = router is confident / tokens collapse to few experts.
    """
    num_blocks = len(hooks)
    fig, ax = plt.subplots(figsize=(12, 5))

    max_entropy = np.log(num_experts)  # entropy of uniform distribution
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, num_blocks))

    for i, hook in enumerate(hooks):
        entropies = []
        for data in hook.routing_data:
            probs = data['probs'].numpy()  # (batch*seq_len, num_experts)
            # Average entropy across all tokens
            eps = 1e-10
            token_entropy = -np.sum(probs * np.log(probs + eps), axis=-1)
            entropies.append(token_entropy.mean())

        ax.plot(entropies, color=colors[i], linewidth=1.5, alpha=0.8,
                label=f'Block {hook.block_idx}')

    ax.axhline(y=max_entropy, color='red', linestyle='--', alpha=0.5,
               label=f'Max entropy (uniform, {max_entropy:.2f})')
    ax.set_xlabel('Sampling step (T → 0)', fontsize=12)
    ax.set_ylabel('Average router entropy (nats)', fontsize=12)
    ax.set_title('Router Entropy Over Denoising Process\n(higher = more uniform routing, lower = more specialized)',
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_spatial_routing(hooks, save_path, num_experts, latent_size):
    """
    Plot 4: Spatial routing map for the LAST timestep of denoising.
    Shows which spatial positions in the image are assigned to which expert.
    Uses the first image in the batch.
    """
    # Take the last routing decision (final denoising step) from the first MoE block
    # Only show first 4 blocks to keep the figure manageable
    show_hooks = hooks[:4]

    num_show = len(show_hooks)
    grid_size = latent_size  # tokens form a grid_size × grid_size grid
    num_tokens = grid_size * grid_size

    fig, axes = plt.subplots(1, num_show, figsize=(4 * num_show, 4))
    if num_show == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap('tab10', num_experts)

    for i, (hook, ax) in enumerate(zip(show_hooks, axes)):
        if not hook.routing_data:
            continue

        # Last timestep, get topk_idx for first image only
        last_data = hook.routing_data[-1]
        topk_idx = last_data['topk_idx'].numpy()  # (batch*seq_len, top_k)

        # Take first image's tokens (first num_tokens entries)
        # The primary expert (highest weight) determines the color
        primary_expert = topk_idx[:num_tokens, 0]  # (num_tokens,)

        spatial_map = primary_expert.reshape(grid_size, grid_size)
        im = ax.imshow(spatial_map, cmap=cmap, vmin=0, vmax=num_experts - 1,
                       interpolation='nearest')
        ax.set_title(f'Block {hook.block_idx}', fontsize=11)
        ax.set_xlabel('Patch column')
        ax.set_ylabel('Patch row')

    # Add shared colorbar
    cbar = plt.colorbar(im, ax=axes, ticks=range(num_experts),
                        label='Primary expert', shrink=0.8)
    fig.suptitle('Spatial Expert Assignment (last denoising step, image 1)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def print_summary_stats(hooks, num_experts):
    """Print text-based summary statistics."""
    print(f"\n{'='*70}")
    print(f"  ROUTING ANALYSIS SUMMARY")
    print(f"{'='*70}")

    for hook in hooks:
        counts = np.zeros(num_experts)
        total = 0
        all_probs = []
        for data in hook.routing_data:
            idx = data['topk_idx'].numpy().flatten()
            for e in range(num_experts):
                counts[e] += (idx == e).sum()
            total += len(idx)
            all_probs.append(data['probs'].numpy())

        fracs = counts / max(total, 1)
        all_probs = np.concatenate(all_probs, axis=0)

        # Compute metrics
        eps = 1e-10
        avg_entropy = -np.sum(all_probs * np.log(all_probs + eps), axis=-1).mean()
        max_entropy = np.log(num_experts)
        entropy_ratio = avg_entropy / max_entropy

        # Load balance: coefficient of variation of expert fractions
        cv = np.std(fracs) / max(np.mean(fracs), eps)

        print(f"\n  Block {hook.block_idx}:")
        print(f"    Expert utilization: {', '.join(f'E{e}={f:.1%}' for e, f in enumerate(fracs))}")
        print(f"    Most used expert:   E{np.argmax(fracs)} ({fracs.max():.1%})")
        print(f"    Least used expert:  E{np.argmin(fracs)} ({fracs.min():.1%})")
        print(f"    Avg router entropy: {avg_entropy:.3f} / {max_entropy:.3f} ({entropy_ratio:.1%} of max)")
        print(f"    Load balance CV:    {cv:.3f} (0 = perfect balance)")

    print(f"\n{'='*70}")


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    latent_size = args.image_size // 8
    moe_blocks = args.moe_blocks

    print(f"\nBuilding DiT-MoE model (MoE blocks: {moe_blocks})...")
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

    # Load pretrained weights
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Handle both formats:
    #   - Vanilla DiT checkpoint: raw state dict (keys are model param names)
    #   - Finetuned MoE checkpoint (trainable-only): keys "ema_trainable_only", "model_trainable_only"
    if isinstance(raw, dict) and "ema_trainable_only" in raw:
        print(f"  -> Finetuned checkpoint (trainable-only) at step={raw.get('train_steps', '?')}.")
        print(f"     Step 1: Loading pretrained backbone from pretrained_models/DiT-XL-2-256x256.pt")
        backbone_state = find_model("pretrained_models/DiT-XL-2-256x256.pt")
        load_pretrained_with_moe(model, backbone_state)
        print(f"     Step 2: Overlaying trained MoE weights (EMA).")
        model.load_state_dict(raw["ema_trainable_only"], strict=False)
    elif isinstance(raw, dict) and "model_trainable_only" in raw:
        print(f"  -> Finetuned checkpoint (trainable-only) at step={raw.get('train_steps', '?')}.")
        print(f"     Step 1: Loading pretrained backbone from pretrained_models/DiT-XL-2-256x256.pt")
        backbone_state = find_model("pretrained_models/DiT-XL-2-256x256.pt")
        load_pretrained_with_moe(model, backbone_state)
        print(f"     Step 2: Overlaying trained MoE weights (model).")
        model.load_state_dict(raw["model_trainable_only"], strict=False)
    elif isinstance(raw, dict) and "ema" in raw:
        print(f"  -> Finetuned checkpoint (full) at step={raw.get('train_steps', '?')}. Loading EMA weights.")
        model.load_state_dict(raw["ema"], strict=False)
    else:
        print(f"  -> Vanilla DiT checkpoint detected. Loading with load_pretrained_with_moe().")
        load_pretrained_with_moe(model, raw)
    model.eval()

    # Install routing hooks
    print("Installing routing hooks...")
    hooks = install_hooks(model, moe_blocks)
    print(f"  Hooks installed on {len(hooks)} MoE blocks")

    # Setup diffusion + VAE
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Sampling setup
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # CFG setup
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Run sampling (routing hooks will capture data)
    print(f"\nRunning sampling ({args.num_sampling_steps} steps)...")
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=True, device=device
    )

    # Decode and save sample image
    samples_img, _ = samples.chunk(2, dim=0)
    samples_img = vae.decode(samples_img / 0.18215).sample
    from torchvision.utils import save_image
    save_image(samples_img, "sample_diffmoe_analyzed.png", nrow=4,
               normalize=True, value_range=(-1, 1))
    print(f"  Saved: sample_diffmoe_analyzed.png")

    # Create output directory
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Generate all visualizations
    num_experts = args.num_experts
    print(f"\nGenerating visualizations in {out_dir}/...")

    plot_expert_utilization_heatmap(
        hooks, os.path.join(out_dir, "expert_utilization_heatmap.png"), num_experts)

    plot_temporal_routing(
        hooks, os.path.join(out_dir, "temporal_routing.png"),
        num_experts, args.num_sampling_steps)

    plot_router_entropy(
        hooks, os.path.join(out_dir, "router_entropy.png"), num_experts)

    plot_spatial_routing(
        hooks, os.path.join(out_dir, "spatial_routing.png"),
        num_experts, latent_size)

    # Print text summary
    print_summary_stats(hooks, num_experts)

    # Restore original forward methods
    for hook in hooks:
        hook.restore()

    print(f"\nDone! All plots saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="routing_analysis",
                        help="Directory to save analysis plots")

    # MoE config
    parser.add_argument("--moe-blocks", type=int, nargs="+", default=[20, 21, 22, 23, 24, 25, 26, 27])
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--n-shared-experts", type=int, default=2)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--no-dwconv", action="store_true")

    args = parser.parse_args()
    main(args)
