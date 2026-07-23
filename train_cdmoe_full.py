"""
train_cdmoe_full.py — Full-scale CD-MoE training (Option B).

This is the paper-ready training script that:
  1. Builds a DiT-MoE model with 8 experts × 8 blocks (matching Diff-MoE)
  2. Replaces ALL standard gates with ConceptAwareMoEGate (CD-MoE loss)
  3. Trains ALL MoE parameters (experts + gates + adaLN + dwconv) — NOT just gates
  4. Keeps pretrained backbone frozen (attention, norms, embedders)
  5. Uses the best hyperparameters from grid search

Direct comparison with Diff-MoE:
  - Same architecture (DiT-XL/2 + MoE in last 8 blocks)
  - Same frozen backbone (pretrained DiT-XL/2)
  - Same training data (ImageNet)
  - DIFFERENT loss: CD-MoE concept-aware loss vs standard load-balance loss

Usage:
  # Full scale (8 experts, blocks 20-27, CD-MoE loss):
  python train_cdmoe_full.py \\
      --data-path /local/a/imagenet/imagenet2012/train/ \\
      --ckpt pretrained_models/DiT-XL-2-256x256.pt \\
      --num-experts 8 \\
      --moe-blocks 20 21 22 23 24 25 26 27 \\
      --diversity-weight 2.0 --collapse-weight 1.0 --gate-temp 0.7 \\
      --epochs 10 --batch-size 16 --lr 1e-4

  # Resume from checkpoint:
  python train_cdmoe_full.py \\
      --data-path /local/a/imagenet/imagenet2012/train/ \\
      --ckpt pretrained_models/DiT-XL-2-256x256.pt \\
      --resume results-cdmoe-full/000-.../checkpoints/0050000.pt \\
      --num-experts 8 --moe-blocks 20 21 22 23 24 25 26 27 \\
      --diversity-weight 2.0 --collapse-weight 1.0 --gate-temp 0.7
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
from PIL import Image
import argparse
import logging
import os
import shutil

from models import DiT_models, load_pretrained_with_moe
from diffusion import create_diffusion
from download import find_model
from moe import DiTBlock_MoE
from moe.concept_gate import ConceptAwareMoEGate


# =============================================================================
#                          Helper Functions
# =============================================================================

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model (trainable params only)."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def center_crop_arr(pil_image, image_size):
    """Center cropping implementation from ADM."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def create_logger(logging_dir):
    """Create a logger that writes to a log file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)


# =============================================================================
#             Gate Replacement (standard → ConceptAwareMoEGate)
# =============================================================================

def replace_all_gates(model, moe_blocks, diversity_weight, collapse_weight,
                      min_expert_util, gate_temp, mode='loss_only'):
    """
    Replace standard MoEGate with ConceptAwareMoEGate in all MoE blocks.
    Copies existing gate weights so routing starts from the same point.
    """
    n_replaced = 0
    for block_idx in moe_blocks:
        block = model.blocks[block_idx]
        if not isinstance(block, DiTBlock_MoE):
            continue

        old_gate = block.moe.gate
        new_gate = ConceptAwareMoEGate(
            embed_dim=old_gate.gating_dim,
            num_experts=old_gate.n_routed_experts,
            num_experts_per_tok=old_gate.top_k,
            aux_loss_alpha=getattr(old_gate, 'alpha', 0.01),
            mode=mode,
            diversity_weight=diversity_weight,
            min_expert_util=min_expert_util,
            collapse_weight=collapse_weight,
            gate_temp=gate_temp,
        )

        # Copy gate weights from old gate (if shapes match)
        device = old_gate.weight.device
        new_gate = new_gate.to(device)
        if old_gate.weight.shape == new_gate.weight.shape:
            new_gate.weight.data.copy_(old_gate.weight.data)

        block.moe.gate = new_gate
        n_replaced += 1

    return n_replaced


# =============================================================================
#                 Freeze / Unfreeze Logic
# =============================================================================

def freeze_pretrained_unfreeze_moe(model, moe_block_indices):
    """
    Freeze the entire pretrained backbone, then unfreeze ALL MoE parameters:
      - .moe (gate + routed experts + shared experts + SE net)
      - .dwconv
      - .adaLN_modulation in MoE blocks
    """
    # Step 1: Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Unfreeze MoE-specific parameters
    trainable_count = 0
    for block_idx in moe_block_indices:
        block = model.blocks[block_idx]
        assert isinstance(block, DiTBlock_MoE), (
            f"Block {block_idx} is not DiTBlock_MoE (got {type(block).__name__})"
        )

        # Unfreeze MoE (gate + all experts)
        for param in block.moe.parameters():
            param.requires_grad = True
            trainable_count += param.numel()

        # Unfreeze adaLN_modulation (re-initialized for MoE)
        for param in block.adaLN_modulation.parameters():
            param.requires_grad = True
            trainable_count += param.numel()

        # Unfreeze dwconv if present
        if hasattr(block, 'dwconv') and block.use_dwconv:
            for param in block.dwconv.parameters():
                param.requires_grad = True
                trainable_count += param.numel()

    return trainable_count


# =============================================================================
#                              Main Training Loop
# =============================================================================

def main(args):
    """Full-scale CD-MoE training."""
    assert torch.cuda.is_available(), "Training requires a GPU."
    device = torch.device('cuda')

    torch.manual_seed(args.global_seed)
    np.random.seed(args.global_seed)

    # ── Setup experiment directory ──────────────────────────────────────
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string = f"CDMoE-full-e{args.num_experts}-b{args.moe_blocks[0]}-{args.moe_blocks[-1]}"
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Arguments: {args}")

    # ── Build model ─────────────────────────────────────────────────────
    latent_size = args.image_size // 8
    moe_blocks = args.moe_blocks

    logger.info(f"Building DiT-MoE: {len(moe_blocks)} MoE blocks, {args.num_experts} experts")
    model = DiT_models['DiT-XL/2-MoE'](
        input_size=latent_size,
        num_classes=args.num_classes,
        moe_blocks=moe_blocks,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        n_shared_experts=args.n_shared_experts,
        rank=args.rank,
        use_dwconv=not args.no_dwconv,
    )

    # ── Load pretrained backbone ────────────────────────────────────────
    logger.info(f"Loading pretrained checkpoint: {args.ckpt}")
    pretrained_state = find_model(args.ckpt)
    load_pretrained_with_moe(model, pretrained_state)

    # ── Replace gates with ConceptAwareMoEGate ──────────────────────────
    n_replaced = replace_all_gates(
        model, moe_blocks,
        diversity_weight=args.diversity_weight,
        collapse_weight=args.collapse_weight,
        min_expert_util=args.min_expert_util,
        gate_temp=args.gate_temp,
        mode=args.mode,
    )
    logger.info(f"Replaced {n_replaced} gates with ConceptAwareMoEGate")
    logger.info(f"  mode={args.mode}, diversity_weight={args.diversity_weight}, "
                f"collapse_weight={args.collapse_weight}, gate_temp={args.gate_temp}")

    # ── Freeze backbone, unfreeze MoE ───────────────────────────────────
    logger.info("Freezing backbone, unfreezing MoE parameters...")
    trainable_count = freeze_pretrained_unfreeze_moe(model, moe_blocks)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    trainable_params_count = sum(p.numel() for p in trainable_params_list)

    logger.info(f"{'='*60}")
    logger.info(f"Parameter summary:")
    logger.info(f"  Total:     {total_params:>15,}")
    logger.info(f"  Trainable: {trainable_params_count:>15,} ({100*trainable_params_count/total_params:.1f}%)")
    logger.info(f"  Frozen:    {total_params - trainable_params_count:>15,}")
    logger.info(f"{'='*60}")

    # ── EMA ──────────────────────────────────────────────────────────────
    ema = deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad = False
    ema.eval()

    # ── Diffusion ────────────────────────────────────────────────────────
    diffusion = create_diffusion(timestep_respacing="")
    logger.info("Diffusion: 1000 steps, linear noise schedule")

    # ── Optimizer ────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=args.weight_decay)
    logger.info(f"Optimizer: AdamW, lr={args.lr}, wd={args.weight_decay}")
    logger.info(f"  Optimizing {len(trainable_params_list)} param groups, "
                f"{trainable_params_count:,} params total")

    # ── Data ─────────────────────────────────────────────────────────────
    logger.info(f"Loading ImageNet from {args.data_path}")
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    logger.info(f"Dataset: {len(dataset):,} samples, {len(loader):,} batches/epoch")

    # ── Resume ───────────────────────────────────────────────────────────
    start_epoch = 0
    train_steps = 0
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        if "model_trainable_only" in resume_ckpt:
            current_state = model.state_dict()
            current_state.update(resume_ckpt["model_trainable_only"])
            model.load_state_dict(current_state)
            ema_state = ema.state_dict()
            ema_state.update(resume_ckpt["ema_trainable_only"])
            ema.load_state_dict(ema_state)
        else:
            model.load_state_dict(resume_ckpt["model"], strict=False)
            ema.load_state_dict(resume_ckpt["ema"], strict=False)

        opt.load_state_dict(resume_ckpt["opt"])
        train_steps = resume_ckpt.get("train_steps", 0)
        start_epoch = resume_ckpt.get("epoch", 0)
        logger.info(f"Resumed at epoch {start_epoch}, step {train_steps}")

    # ── Training loop ────────────────────────────────────────────────────
    update_ema(ema, model, decay=0)  # Initialize EMA
    model.train()

    log_steps = 0
    running_loss = 0.0
    start_time = time()

    logger.info(f"Starting training...")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Steps per epoch: {len(loader):,}")
    logger.info(f"  Total steps: ~{len(loader) * args.epochs:,}")
    if args.max_steps > 0:
        logger.info(f"  Max steps: {args.max_steps} (early stop)")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Beginning epoch {epoch}/{args.epochs - 1}...")
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            # Encode to latent space
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            # Store class labels for concept-aware loss
            model._current_class_labels = y

            # Diffusion training
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs=dict(y=y))
            loss = loss_dict["loss"].mean()

            opt.zero_grad()
            loss.backward()

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params_list, args.max_grad_norm)

            opt.step()
            update_ema(ema, model)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            # Logging
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                elapsed = time() - start_time
                steps_per_sec = log_steps / elapsed
                avg_loss = running_loss / log_steps
                gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9
                logger.info(
                    f"(step={train_steps:07d}, epoch={epoch}) "
                    f"Train Loss: {avg_loss:.4f}, "
                    f"Steps/Sec: {steps_per_sec:.2f}, "
                    f"GPU Mem: {gpu_mem:.1f}GB"
                )

                # Log gate weight norms + expert utilization every 1000 steps
                if train_steps % 1000 == 0:
                    for idx in moe_blocks:
                        block = model.blocks[idx]
                        if hasattr(block.moe, 'gate') and isinstance(block.moe.gate, ConceptAwareMoEGate):
                            gate = block.moe.gate
                            w_norms = gate.weight.data.norm(dim=1)
                            norm_str = ", ".join(f"E{e}={w_norms[e]:.2f}"
                                                 for e in range(gate.n_routed_experts))
                            logger.info(f"  Block {idx} gate norms: {norm_str}")

                running_loss = 0.0
                log_steps = 0
                start_time = time()

            # Checkpoint (trainable-only, save to /tmp first)
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
                model_state = {k: v for k, v in model.state_dict().items() if k in trainable_names}
                ema_state = {k: v for k, v in ema.state_dict().items() if k in trainable_names}

                ckpt = {
                    "model_trainable_only": model_state,
                    "ema_trainable_only": ema_state,
                    "opt": opt.state_dict(),
                    "args": args,
                    "train_steps": train_steps,
                    "epoch": epoch,
                }

                ckpt_filename = f"{train_steps:07d}.pt"
                tmp_path = f"/tmp/{ckpt_filename}"
                final_path = f"{checkpoint_dir}/{ckpt_filename}"

                torch.save(ckpt, tmp_path)
                size_mb = os.path.getsize(tmp_path) / 1e6
                logger.info(f"Saved checkpoint to /tmp ({size_mb:.0f} MB)")

                try:
                    shutil.move(tmp_path, final_path)
                    logger.info(f"Moved to: {final_path}")
                except Exception as e:
                    logger.warning(f"Move failed: {e}. Checkpoint safe at {tmp_path}")

            # Early stop
            if args.max_steps > 0 and train_steps >= args.max_steps:
                logger.info(f"Reached max_steps={args.max_steps}, stopping.")
                break

        if args.max_steps > 0 and train_steps >= args.max_steps:
            break

    # ── Save final checkpoint ───────────────────────────────────────────
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    model_state = {k: v for k, v in model.state_dict().items() if k in trainable_names}
    ema_state = {k: v for k, v in ema.state_dict().items() if k in trainable_names}

    final_ckpt = {
        "model_trainable_only": model_state,
        "ema_trainable_only": ema_state,
        "opt": opt.state_dict(),
        "args": args,
        "train_steps": train_steps,
        "epoch": args.epochs,
    }
    tmp_path = "/tmp/final_cdmoe_full.pt"
    final_path = f"{checkpoint_dir}/final.pt"
    torch.save(final_ckpt, tmp_path)
    logger.info(f"Saved final checkpoint to /tmp")
    try:
        shutil.move(tmp_path, final_path)
        logger.info(f"Moved final checkpoint to: {final_path}")
    except Exception as e:
        logger.warning(f"Move failed: {e}. Checkpoint safe at {tmp_path}")

    logger.info(f"Done! Total training steps: {train_steps:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-scale CD-MoE Training")

    # Data
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")

    # Model
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Pretrained DiT-XL/2 checkpoint")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a CD-MoE full training checkpoint")
    parser.add_argument("--moe-blocks", type=int, nargs="+", default=list(range(20, 28)),
                        help="Block indices for MoE (default: 20-27)")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--n-shared-experts", type=int, default=2)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--no-dwconv", action="store_true")

    # CD-MoE loss hyperparameters (use grid search best values as defaults)
    parser.add_argument("--mode", type=str, default="loss_only",
                        choices=["standard", "loss_only", "full"])
    parser.add_argument("--diversity-weight", type=float, default=2.0)
    parser.add_argument("--collapse-weight", type=float, default=1.0)
    parser.add_argument("--min-expert-util", type=float, default=0.5)
    parser.add_argument("--gate-temp", type=float, default=0.7)

    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (may need to reduce for 8-expert memory)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)

    # Logging
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--results-dir", type=str, default="results-cdmoe-full")

    args = parser.parse_args()
    main(args)
