"""
sample_cdmoe.py — Generate images from CD-MoE model for visual + FID comparison.

This script:
1. Loads base Diff-MoE model + overlays CD-MoE trained gate weights
2. Generates sample images (visual grid for quick comparison)
3. Optionally generates N images per class for FID evaluation

Usage:
    # Quick visual comparison (8 classes, same seed):
    python sample_cdmoe.py \
        --base-ckpt results-finetune/004-.../0115000.pt \
        --cdmoe-ckpt results-cdmoe/003-.../final.pt \
        --output-dir samples_cdmoe_v2 \
        --num-sampling-steps 250

    # Baseline comparison (no CD-MoE gates, same seed):
    python sample_cdmoe.py \
        --base-ckpt results-finetune/004-.../0115000.pt \
        --output-dir samples_baseline \
        --num-sampling-steps 250

    # FID evaluation (generate many images):
    python sample_cdmoe.py \
        --base-ckpt results-finetune/004-.../0115000.pt \
        --cdmoe-ckpt results-cdmoe/003-.../final.pt \
        --output-dir samples_cdmoe_v2_fid \
        --fid-mode --fid-num-per-class 50 --fid-num-classes 100 \
        --num-sampling-steps 250
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import os
import json
import numpy as np
from pathlib import Path
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models, load_pretrained_with_moe
from PIL import Image


# ImageNet class names for the 8 test classes
CLASS_NAMES = {
    207: "Golden Retriever", 360: "Otter", 387: "Red Panda",
    974: "Geyser", 88: "Macaw", 979: "Valley", 417: "Balloon", 279: "Arctic Fox",
}


def load_model(args, device):
    """Build model and load base checkpoint + optional CD-MoE gates."""
    latent_size = args.image_size // 8
    moe_blocks = args.moe_blocks

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

    # Load base checkpoint
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

    # Optionally overlay CD-MoE gate weights
    if args.cdmoe_ckpt:
        print(f"  Loading CD-MoE gate weights: {args.cdmoe_ckpt}")
        cdmoe = torch.load(args.cdmoe_ckpt, map_location="cpu", weights_only=False)
        print(f"  → Mode: {cdmoe.get('mode')}, Steps: {cdmoe.get('train_steps')}")
        for key, state in cdmoe['gate_state'].items():
            block_idx = int(key.split('.')[1])
            gate = model.blocks[block_idx].moe.gate
            gate.weight.data.copy_(state['weight'].to(device))
            print(f"  → Loaded gate for block {block_idx}")
    else:
        print("  → No CD-MoE gates (baseline mode)")

    model.eval()
    return model


def sample_images(model, diffusion, vae, class_labels, args, device, seed=None):
    """Generate images for given class labels. Returns decoded image tensors."""
    if seed is not None:
        torch.manual_seed(seed)

    latent_size = args.image_size // 8
    n = len(class_labels)

    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Classifier-free guidance
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample
    return samples


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    model_type = "CD-MoE" if args.cdmoe_ckpt else "Baseline Diff-MoE"
    print(f"\n{'='*60}")
    print(f"  IMAGE GENERATION — {model_type}")
    print(f"{'='*60}")

    model = load_model(args, device)
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    if args.fid_mode:
        # ── FID mode: generate many images per class ──────────────────
        print(f"\n  FID mode: {args.fid_num_per_class} images × {args.fid_num_classes} classes")
        print(f"  Total: {args.fid_num_per_class * args.fid_num_classes} images")

        # Select class labels (first N classes, or random)
        if args.fid_classes:
            all_classes = args.fid_classes
        else:
            rng = np.random.RandomState(42)
            all_classes = rng.choice(1000, args.fid_num_classes, replace=False).tolist()

        img_dir = os.path.join(args.output_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        total_generated = 0
        batch_size = min(args.fid_batch_size, args.fid_num_per_class)

        for cls_idx, cls_label in enumerate(all_classes):
            cls_name = CLASS_NAMES.get(cls_label, f"class_{cls_label}")
            remaining = args.fid_num_per_class
            img_count = 0

            while remaining > 0:
                n_batch = min(batch_size, remaining)
                labels = [cls_label] * n_batch

                samples = sample_images(
                    model, diffusion, vae, labels, args, device,
                    seed=args.seed + total_generated
                )

                for i in range(n_batch):
                    img = samples[i]
                    img = (img + 1) / 2  # [-1,1] → [0,1]
                    img = img.clamp(0, 1)
                    img_pil = Image.fromarray(
                        (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    )
                    img_pil.save(os.path.join(img_dir, f"{cls_label:04d}_{img_count:04d}.png"))
                    img_count += 1
                    total_generated += 1

                remaining -= n_batch

            if (cls_idx + 1) % 10 == 0 or (cls_idx + 1) == len(all_classes):
                print(f"  [{cls_idx+1}/{len(all_classes)}] Generated {total_generated} images total")

        # Save metadata
        meta = {
            "model_type": model_type,
            "base_ckpt": args.base_ckpt,
            "cdmoe_ckpt": args.cdmoe_ckpt,
            "num_classes": len(all_classes),
            "num_per_class": args.fid_num_per_class,
            "total_images": total_generated,
            "sampling_steps": args.num_sampling_steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "class_labels": all_classes,
        }
        with open(os.path.join(args.output_dir, "generation_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n  Total: {total_generated} images saved to {img_dir}/")
        print(f"  Run FID: python -m pytorch_fid {img_dir} <reference_stats_path>")

    else:
        # ── Visual comparison mode ────────────────────────────────────
        class_labels = args.class_labels
        print(f"\n  Generating {len(class_labels)} images ({args.num_sampling_steps} steps)...")
        print(f"  Classes: {[CLASS_NAMES.get(c, c) for c in class_labels]}")
        print(f"  Seed: {args.seed}")

        samples = sample_images(model, diffusion, vae, class_labels, args, device, seed=args.seed)

        # Save grid
        grid_path = os.path.join(args.output_dir, "sample_grid.png")
        save_image(samples, grid_path, nrow=4, normalize=True, value_range=(-1, 1))
        print(f"  Saved grid: {grid_path}")

        # Save individual images with class names
        for i, cls_label in enumerate(class_labels):
            cls_name = CLASS_NAMES.get(cls_label, f"class_{cls_label}")
            img = samples[i]
            img_path = os.path.join(args.output_dir, f"{cls_name.replace(' ', '_').lower()}.png")
            save_image(img, img_path, normalize=True, value_range=(-1, 1))

        print(f"  Saved {len(class_labels)} individual images to {args.output_dir}/")

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--base-ckpt", type=str, required=True)
    parser.add_argument("--cdmoe-ckpt", type=str, default=None,
                        help="CD-MoE gate checkpoint. Omit for baseline.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae", type=str, default="mse")
    parser.add_argument("--output-dir", type=str, default="samples_comparison")

    # MoE config
    parser.add_argument("--moe-blocks", type=int, nargs="+", default=[24, 25, 26, 27])
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--n-shared-experts", type=int, default=2)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--no-dwconv", action="store_true")

    # Visual comparison mode
    parser.add_argument("--class-labels", type=int, nargs="+",
                        default=[207, 360, 387, 974, 88, 979, 417, 279])

    # FID mode
    parser.add_argument("--fid-mode", action="store_true",
                        help="Generate many images for FID evaluation")
    parser.add_argument("--fid-num-per-class", type=int, default=50,
                        help="Images to generate per class for FID")
    parser.add_argument("--fid-num-classes", type=int, default=100,
                        help="Number of classes to generate for FID")
    parser.add_argument("--fid-batch-size", type=int, default=8,
                        help="Batch size for FID generation")
    parser.add_argument("--fid-classes", type=int, nargs="+", default=None,
                        help="Specific class labels for FID (default: random 100)")

    args = parser.parse_args()
    main(args)
