# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pretrained DiT with Diff-MoE blocks.

Loads a vanilla DiT-XL/2 checkpoint, replaces selected blocks with the
Diff-MoE architecture (routed experts + shared expert + timestep conditioning),
and runs DDPM sampling with classifier-free guidance.

NOTE: Since MoE experts/router are randomly initialized (not finetuned),
the output quality from MoE blocks will differ from the vanilla DiT baseline.
This script verifies the architecture runs correctly end-to-end.
To get publication-quality results, finetune the MoE blocks.

Usage:
    python sample_diffmoe.py --image-size 256 --num-sampling-steps 250
    python sample_diffmoe.py --image-size 256 --num-sampling-steps 50 --moe-blocks 24 25 26 27
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models, load_pretrained_with_moe
import argparse


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2-MoE", "Only DiT-XL/2-MoE is available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # ============================================================
    # Build DiT-MoE model
    # ============================================================
    latent_size = args.image_size // 8
    moe_blocks = args.moe_blocks

    print(f"\nBuilding DiT-MoE model...")
    print(f"  MoE blocks: {moe_blocks}")
    print(f"  Num experts: {args.num_experts}")
    print(f"  Top-k: {args.num_experts_per_tok}")
    print(f"  Shared experts: {args.n_shared_experts}")

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        moe_blocks=moe_blocks,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        n_shared_experts=args.n_shared_experts,
        rank=args.rank,
        use_dwconv=not args.no_dwconv,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ============================================================
    # Load pretrained DiT checkpoint (attention + norms from vanilla DiT,
    # MoE params stay randomly initialized)
    # ============================================================
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    print(f"\nLoading checkpoint: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Handle both formats:
    #   - Vanilla DiT checkpoint: raw state dict (keys are model param names)
    #   - Finetuned MoE checkpoint (trainable-only): keys are "ema_trainable_only", "model_trainable_only"
    if isinstance(raw, dict) and "ema_trainable_only" in raw:
        # Step 1: load pretrained DiT backbone (attention, norms, embedders)
        print(f"  -> Finetuned MoE checkpoint (trainable-only) at step={raw.get('train_steps', '?')}.")
        print(f"     Step 1: Loading pretrained backbone from pretrained_models/DiT-XL-2-256x256.pt")
        backbone_state = find_model("pretrained_models/DiT-XL-2-256x256.pt")
        load_pretrained_with_moe(model, backbone_state)
        # Step 2: overlay the trained MoE weights on top
        print(f"     Step 2: Overlaying trained MoE weights (EMA).")
        model.load_state_dict(raw["ema_trainable_only"], strict=False)
    elif isinstance(raw, dict) and "model_trainable_only" in raw:
        print(f"  -> Finetuned MoE checkpoint (trainable-only) at step={raw.get('train_steps', '?')}.")
        print(f"     Step 1: Loading pretrained backbone from pretrained_models/DiT-XL-2-256x256.pt")
        backbone_state = find_model("pretrained_models/DiT-XL-2-256x256.pt")
        load_pretrained_with_moe(model, backbone_state)
        print(f"     Step 2: Overlaying trained MoE weights (model).")
        model.load_state_dict(raw["model_trainable_only"], strict=False)
    elif isinstance(raw, dict) and "ema" in raw:
        print(f"  -> Finetuned MoE checkpoint (full) at step={raw.get('train_steps', '?')}. Loading EMA weights.")
        model.load_state_dict(raw["ema"], strict=False)
    else:
        # Vanilla DiT checkpoint — use the original pretrained loader
        print(f"  -> Vanilla DiT checkpoint detected. Loading with load_pretrained_with_moe().")
        load_pretrained_with_moe(model, raw)

    model.eval()  # important!

    # ============================================================
    # Diffusion + VAE
    # ============================================================
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # ============================================================
    # Labels to condition the model with (feel free to change)
    # ============================================================
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # ============================================================
    # Sample images
    # ============================================================
    print(f"\nStarting diffusion sampling ({args.num_sampling_steps} steps)...")
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    output_path = args.output or "sample_diffmoe.png"
    save_image(samples, output_path, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiT-XL/2-MoE",
                        help="Model config name")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to a DiT checkpoint (default: auto-download DiT-XL/2).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: sample_diffmoe.png)")

    # MoE-specific arguments
    parser.add_argument("--moe-blocks", type=int, nargs="+", default=[20, 21, 22, 23, 24, 25, 26, 27],
                        help="Which block indices to convert to MoE (default: last 8)")
    parser.add_argument("--num-experts", type=int, default=8,
                        help="Number of routed experts per MoE block")
    parser.add_argument("--num-experts-per-tok", type=int, default=2,
                        help="Top-k experts selected per token")
    parser.add_argument("--n-shared-experts", type=int, default=2,
                        help="Number of always-active shared experts")
    parser.add_argument("--rank", type=int, default=64,
                        help="Low-rank bottleneck dim for expert adaLN")
    parser.add_argument("--no-dwconv", action="store_true",
                        help="Disable depthwise conv before MoE")

    args = parser.parse_args()
    main(args)
