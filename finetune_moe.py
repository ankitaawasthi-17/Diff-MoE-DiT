# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Finetune only the MoE components of a DiT-MoE model, keeping the pretrained
backbone (attention layers, layer norms, embedders, final layer) frozen.

This script:
  1. Builds a DiT_MoE model with the specified MoE block indices.
  2. Loads a pretrained vanilla DiT-XL/2 checkpoint into it (attention/norms
     load normally; MoE-specific params stay randomly initialized).
  3. Freezes everything EXCEPT:
       - `.moe` sub-modules in MoE blocks (gate, routed experts, shared experts, SE net)
       - `.dwconv` in MoE blocks (if use_dwconv=True)
       - `.adaLN_modulation` in MoE blocks (re-initialized, so needs training)
  4. Trains with the standard diffusion loss on ImageNet latents or images.

Supports two data modes:
  --use-latents   Load pre-extracted .pt latent files (faster, no VAE needed).
                  Each .pt file should contain a dict {'latent': Tensor, 'label': int}.
  (default)       Load raw ImageNet images and encode them on-the-fly with the
                  Stable Diffusion VAE.

Usage examples:
  # Finetune with pre-extracted latents (recommended for speed):
  python finetune_moe.py \\
      --data-path /path/to/imagenet_latents/ \\
      --use-latents \\
      --ckpt DiT-XL-2-256x256.pt \\
      --moe-blocks 20 21 22 23 24 25 26 27 \\
      --num-experts 8 \\
      --epochs 10 \\
      --batch-size 32 \\
      --lr 1e-4

  # Finetune with raw ImageNet images (slower, needs VAE):
  python finetune_moe.py \\
      --data-path /path/to/imagenet/train/ \\
      --ckpt DiT-XL-2-256x256.pt \\
      --moe-blocks 20 21 22 23 24 25 26 27 \\
      --image-size 256 \\
      --vae ema

  # Custom MoE config with more experts:
  python finetune_moe.py \\
      --data-path /path/to/imagenet_latents/ \\
      --use-latents \\
      --ckpt DiT-XL-2-256x256.pt \\
      --moe-blocks 14 15 16 17 18 19 20 21 22 23 24 25 26 27 \\
      --num-experts 16 \\
      --num-experts-per-tok 4 \\
      --n-shared-experts 2 \\
      --rank 64
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
import sys
import shutil

from models import DiT_models, load_pretrained_with_moe
from diffusion import create_diffusion
from download import find_model
from moe import DiTBlock_MoE


# =============================================================================
#                          Helper Functions
# =============================================================================

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    Only updates parameters that require grad in the source model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
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
    logger = logging.getLogger(__name__)
    return logger


# =============================================================================
#                      Pre-extracted Latent Dataset
# =============================================================================

class LatentDataset(Dataset):
    """
    Dataset for pre-extracted VAE latent files.

    Expects a directory containing .pt files. Each file should be a dict:
        {'latent': Tensor of shape (4, H, W), 'label': int}

    Alternatively, supports .npz format:
        Contains 'latent' (numpy array) and 'label' (int).
    """
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        # Collect all .pt and .npz files
        self.files = sorted(
            glob(os.path.join(data_dir, '**', '*.pt'), recursive=True) +
            glob(os.path.join(data_dir, '**', '*.npz'), recursive=True)
        )
        if len(self.files) == 0:
            raise ValueError(
                f"No .pt or .npz files found in {data_dir}. "
                f"Expected pre-extracted latent files."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        if fpath.endswith('.pt'):
            data = torch.load(fpath, map_location='cpu', weights_only=False)
            latent = data['latent']  # (4, H, W) tensor
            label = int(data['label'])
        elif fpath.endswith('.npz'):
            data = np.load(fpath)
            latent = torch.from_numpy(data['latent']).float()
            label = int(data['label'])
        else:
            raise ValueError(f"Unsupported file format: {fpath}")
        return latent, label


# =============================================================================
#                    Freeze / Unfreeze Logic
# =============================================================================

def freeze_pretrained_params(model, moe_block_indices):
    """
    Freeze the entire model, then selectively unfreeze MoE-specific parameters
    in the designated MoE blocks.

    Frozen (pretrained backbone):
      - x_embedder, t_embedder, y_embedder, pos_embed
      - final_layer
      - All standard DiTBlock parameters
      - Attention (.attn) and norms (.norm1, .norm2) in MoE blocks

    Unfrozen (trainable, randomly initialized MoE components):
      - .moe sub-modules in MoE blocks (gate, routed experts, shared experts, SE net)
      - .dwconv in MoE blocks (spatial mixing before MoE)
      - .adaLN_modulation in MoE blocks (re-initialized for MoE conditioning)
    """
    # Step 1: Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Unfreeze MoE-specific parameters in designated blocks
    for block_idx in moe_block_indices:
        block = model.blocks[block_idx]
        assert isinstance(block, DiTBlock_MoE), (
            f"Block {block_idx} is not a DiTBlock_MoE (got {type(block).__name__}). "
            f"Check --moe-blocks argument."
        )

        # Unfreeze the MoE sub-module (gate, all experts, shared experts, SE net)
        for param in block.moe.parameters():
            param.requires_grad = True

        # Unfreeze the adaLN_modulation (re-initialized / zero-initialized for MoE)
        for param in block.adaLN_modulation.parameters():
            param.requires_grad = True

        # Unfreeze dwconv if present
        if hasattr(block, 'dwconv') and block.use_dwconv:
            for param in block.dwconv.parameters():
                param.requires_grad = True


def log_param_stats(model, logger):
    """Log total, trainable, and frozen parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logger.info(f"{'='*60}")
    logger.info(f"Parameter summary:")
    logger.info(f"  Total parameters:     {total_params:>15,}")
    logger.info(f"  Trainable parameters: {trainable_params:>15,} ({100*trainable_params/total_params:.1f}%)")
    logger.info(f"  Frozen parameters:    {frozen_params:>15,} ({100*frozen_params/total_params:.1f}%)")
    logger.info(f"{'='*60}")

    # Detailed breakdown by module type
    moe_params = 0
    dwconv_params = 0
    adaln_moe_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if '.moe.' in name:
                moe_params += p.numel()
            elif '.dwconv.' in name:
                dwconv_params += p.numel()
            elif '.adaLN_modulation.' in name:
                adaln_moe_params += p.numel()

    logger.info(f"Trainable parameter breakdown:")
    logger.info(f"  MoE (gate+experts+SE):    {moe_params:>12,}")
    logger.info(f"  DWConv:                   {dwconv_params:>12,}")
    logger.info(f"  adaLN (MoE blocks):       {adaln_moe_params:>12,}")
    logger.info(f"{'='*60}")


# =============================================================================
#                              Main Training Loop
# =============================================================================

def main(args):
    """Finetune MoE components of a pretrained DiT model on a single GPU."""
    assert torch.cuda.is_available(), "Training requires at least one GPU."
    device = torch.device('cuda')

    # Seed for reproducibility
    torch.manual_seed(args.global_seed)
    np.random.seed(args.global_seed)

    # -------------------------------------------------------------------------
    # Setup experiment directory
    # -------------------------------------------------------------------------
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")
    moe_str = f"moe{args.moe_blocks[0]}-{args.moe_blocks[-1]}"
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-finetune-{moe_str}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Arguments: {args}")

    # -------------------------------------------------------------------------
    # Build DiT-MoE model and load pretrained weights
    # -------------------------------------------------------------------------
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    logger.info(f"Building {args.model} with MoE blocks at indices {args.moe_blocks}...")
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        moe_blocks=args.moe_blocks,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        n_shared_experts=args.n_shared_experts,
        rank=args.rank,
        use_dwconv=not args.no_dwconv,
    )

    # Load pretrained DiT checkpoint
    logger.info(f"Loading pretrained checkpoint: {args.ckpt}")
    pretrained_state_dict = find_model(args.ckpt)
    load_pretrained_with_moe(model, pretrained_state_dict)

    # -------------------------------------------------------------------------
    # Freeze backbone, unfreeze MoE params
    # -------------------------------------------------------------------------
    logger.info("Freezing pretrained backbone, unfreezing MoE parameters...")
    freeze_pretrained_params(model, args.moe_blocks)
    model = model.to(device)
    log_param_stats(model, logger)

    # Create EMA copy (tracks only trainable params effectively)
    ema = deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad = False
    ema.eval()

    # -------------------------------------------------------------------------
    # Setup diffusion
    # -------------------------------------------------------------------------
    diffusion = create_diffusion(timestep_respacing="")  # 1000 steps, linear schedule
    logger.info("Diffusion: 1000 steps, linear noise schedule")

    # -------------------------------------------------------------------------
    # Setup optimizer (only trainable parameters)
    # -------------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    logger.info(f"Optimizer: AdamW, lr={args.lr}, weight_decay={args.weight_decay}")
    logger.info(f"  Optimizing {len(trainable_params)} parameter groups, "
                f"{sum(p.numel() for p in trainable_params):,} parameters total")

    # -------------------------------------------------------------------------
    # Setup data
    # -------------------------------------------------------------------------
    if args.use_latents:
        # Pre-extracted latent files (no VAE needed)
        logger.info(f"Loading pre-extracted latents from {args.data_path}")
        dataset = LatentDataset(args.data_path)
        vae = None
    else:
        # Raw images: need VAE for on-the-fly encoding
        logger.info(f"Loading ImageNet images from {args.data_path}")
        logger.info(f"Will encode on-the-fly with SD-VAE-ft-{args.vae}")
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
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
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset: {len(dataset):,} samples, {len(loader):,} batches/epoch")

    # -------------------------------------------------------------------------
    # Optionally resume from a finetuning checkpoint
    # -------------------------------------------------------------------------
    start_epoch = 0
    train_steps = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        # Support BOTH checkpoint formats:
        #   - OLD format: full "model"/"ema" state dicts (today's earlier checkpoints)
        #   - NEW format: "model_trainable_only"/"ema_trainable_only" (this fix)
        # This lets you resume from an old-format checkpoint today and
        # all checkpoints produced FROM NOW ON will be the smaller new format.
        if "model_trainable_only" in resume_ckpt:
            logger.info("Detected NEW checkpoint format (trainable-only). Merging onto pretrained backbone.")
            current_model_state = model.state_dict()
            current_model_state.update(resume_ckpt["model_trainable_only"])
            model.load_state_dict(current_model_state)

            current_ema_state = ema.state_dict()
            current_ema_state.update(resume_ckpt["ema_trainable_only"])
            ema.load_state_dict(current_ema_state)
        else:
            logger.info("Detected OLD checkpoint format (full state dict). Loading directly.")
            model.load_state_dict(resume_ckpt["model"])
            ema.load_state_dict(resume_ckpt["ema"])

        opt.load_state_dict(resume_ckpt["opt"])
        train_steps = resume_ckpt.get("train_steps", 0)
        start_epoch = resume_ckpt.get("epoch", 0)
        logger.info(f"Resumed at epoch {start_epoch}, step {train_steps}")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    update_ema(ema, model, decay=0)  # Initialize EMA with current weights
    model.train()

    log_steps = 0
    running_loss = 0.0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs ({len(loader) * args.epochs} steps)...")
    logger.info(f"Logging every {args.log_every} steps, checkpointing every {args.ckpt_every} steps")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Beginning epoch {epoch}/{args.epochs - 1}...")
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            # Encode images to latents if not using pre-extracted latents
            if not args.use_latents:
                with torch.no_grad():
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            # Sample random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            # Compute diffusion training loss
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            # Backward pass and optimizer step
            opt.zero_grad()
            loss.backward()

            # Optional gradient clipping for stability
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

            opt.step()
            update_ema(ema, model)

            # Logging
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                logger.info(
                    f"(step={train_steps:07d}, epoch={epoch}) "
                    f"Train Loss: {avg_loss:.4f}, "
                    f"Steps/Sec: {steps_per_sec:.2f}, "
                    f"GPU Mem: {torch.cuda.max_memory_allocated(device) / 1e9:.1f}GB"
                )
                running_loss = 0.0
                log_steps = 0
                start_time = time()

            # Save checkpoint
            # FIX: write to LOCAL disk (/tmp) first, then move into place.
            # The repeated crashes ("iostream error", ".nfs... Device or
            # resource busy") happen because this checkpoint folder lives
            # on an NFS network filesystem, and writing a large (~10GB)
            # file directly to NFS is prone to write races/interruptions.
            # Writing to /tmp (local disk, no network involved) first
            # avoids that failure mode entirely. The move-into-place step
            # at the end is just a file copy on this system (since /tmp
            # and the NFS mount are different filesystems), but that
            # copy is far less failure-prone than a direct large write
            # over NFS, and even if the move itself hiccups, the
            # checkpoint already exists safely in /tmp as a fallback.
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                # FIX: only save TRAINABLE params (MoE blocks), not the
                # full ~927M-param model. The frozen backbone is already
                # on disk in the original pretrained checkpoint, so
                # re-saving it every 5000 steps wastes ~7GB per save for
                # no reason. This also reduces checkpoint size from
                # ~10GB to ~3GB, which should make this save far less
                # likely to hit whatever this NFS/tmpfs failure mode is
                # (it's recurred twice today at the exact same byte
                # offset despite the local-tmp-then-move fix, suggesting
                # a size-sensitive edge case rather than pure disk space).
                trainable_param_names = {
                    name for name, p in model.named_parameters() if p.requires_grad
                }
                full_state = model.state_dict()
                trainable_state = {
                    k: v for k, v in full_state.items() if k in trainable_param_names
                }
                ema_full_state = ema.state_dict()
                ema_trainable_state = {
                    k: v for k, v in ema_full_state.items() if k in trainable_param_names
                }

                checkpoint = {
                    "model_trainable_only": trainable_state,
                    "ema_trainable_only": ema_trainable_state,
                    "opt": opt.state_dict(),
                    "args": args,
                    "train_steps": train_steps,
                    "epoch": epoch,
                }
                checkpoint_filename = f"{train_steps:07d}.pt"
                tmp_path = f"/tmp/{checkpoint_filename}"
                final_dest_path = f"{checkpoint_dir}/{checkpoint_filename}"

                torch.save(checkpoint, tmp_path)
                ckpt_size_mb = os.path.getsize(tmp_path) / 1e6
                logger.info(f"Saved checkpoint to local tmp: {tmp_path} ({ckpt_size_mb:.0f} MB, trainable-only)")

                try:
                    shutil.move(tmp_path, final_dest_path)
                    logger.info(f"Moved checkpoint to: {final_dest_path}")
                except Exception as e:
                    # If the move fails for any reason, the checkpoint is
                    # STILL SAFE in /tmp -- log this clearly so it's not
                    # silently lost, rather than crashing the whole run.
                    logger.warning(
                        f"WARNING: failed to move checkpoint from {tmp_path} "
                        f"to {final_dest_path}: {e}. "
                        f"The checkpoint is still safely saved at {tmp_path} -- "
                        f"copy it manually from there if needed."
                    )

    # Save final checkpoint (same trainable-only + local-tmp-then-move pattern)
    trainable_param_names = {
        name for name, p in model.named_parameters() if p.requires_grad
    }
    full_state = model.state_dict()
    trainable_state = {k: v for k, v in full_state.items() if k in trainable_param_names}
    ema_full_state = ema.state_dict()
    ema_trainable_state = {k: v for k, v in ema_full_state.items() if k in trainable_param_names}

    final_checkpoint = {
        "model_trainable_only": trainable_state,
        "ema_trainable_only": ema_trainable_state,
        "opt": opt.state_dict(),
        "args": args,
        "train_steps": train_steps,
        "epoch": args.epochs,
    }
    tmp_final_path = "/tmp/final.pt"
    final_path = f"{checkpoint_dir}/final.pt"
    torch.save(final_checkpoint, tmp_final_path)
    logger.info(f"Saved final checkpoint to local tmp: {tmp_final_path}")
    try:
        shutil.move(tmp_final_path, final_path)
        logger.info(f"Moved final checkpoint to: {final_path}")
    except Exception as e:
        logger.warning(
            f"WARNING: failed to move final checkpoint from {tmp_final_path} "
            f"to {final_path}: {e}. "
            f"The checkpoint is still safely saved at {tmp_final_path}."
        )
    logger.info(f"Done! Total training steps: {train_steps:,}")


# =============================================================================
#                              Argument Parser
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune MoE blocks of a pretrained DiT model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With pre-extracted latents:
  python finetune_moe.py --data-path /data/imagenet_latents --use-latents --ckpt DiT-XL-2-256x256.pt

  # With raw images:
  python finetune_moe.py --data-path /data/imagenet/train --ckpt DiT-XL-2-256x256.pt --image-size 256

  # Custom MoE config:
  python finetune_moe.py --data-path /data/imagenet_latents --use-latents \\
      --ckpt DiT-XL-2-256x256.pt --moe-blocks 14 15 16 17 18 19 20 21 22 23 24 25 26 27 \\
      --num-experts 16 --num-experts-per-tok 4
        """
    )

    # Data arguments
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to ImageNet images or pre-extracted latent directory.")
    parser.add_argument("--use-latents", action="store_true",
                        help="If set, load pre-extracted .pt/.npz latent files instead of raw images.")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256,
                        help="Image resolution (only used with raw images).")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema",
                        help="Which SD VAE to use for on-the-fly encoding.")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="Number of classes in the dataset.")

    # Model / checkpoint arguments
    parser.add_argument("--model", type=str, default="DiT-XL/2-MoE",
                        choices=list(DiT_models.keys()),
                        help="Model architecture to use.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path or name of pretrained DiT checkpoint "
                             "(e.g. 'DiT-XL-2-256x256.pt' or a local path).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a finetuning checkpoint to resume from.")

    # MoE architecture arguments
    parser.add_argument("--moe-blocks", type=int, nargs="+", default=list(range(20, 28)),
                        help="Block indices to convert to MoE (default: 20-27, i.e. last 8 of 28).")
    parser.add_argument("--num-experts", type=int, default=8,
                        help="Number of routed experts per MoE block.")
    parser.add_argument("--num-experts-per-tok", type=int, default=2,
                        help="Top-k experts selected per token.")
    parser.add_argument("--n-shared-experts", type=int, default=2,
                        help="Number of shared (always-active) experts.")
    parser.add_argument("--rank", type=int, default=64,
                        help="Low-rank bottleneck dimension for expert adaLN.")
    parser.add_argument("--no-dwconv", action="store_true",
                        help="Disable depthwise conv before MoE layer.")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for AdamW optimizer.")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay for AdamW optimizer.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = no clipping).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Per-GPU batch size.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--global-seed", type=int, default=0,
                        help="Random seed for reproducibility.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of DataLoader workers.")

    # Logging and checkpointing
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log training loss every N steps.")
    parser.add_argument("--ckpt-every", type=int, default=5000,
                        help="Save checkpoint every N steps.")
    parser.add_argument("--results-dir", type=str, default="results-finetune",
                        help="Directory to store experiment results.")

    args = parser.parse_args()
    main(args)