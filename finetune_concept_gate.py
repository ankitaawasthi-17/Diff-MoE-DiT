"""
finetune_concept_gate.py
========================
Stage 2 training: Finetune ONLY the concept-aware gate weights.

This script:
1. Loads a pre-trained Diff-MoE model (backbone + standard MoE experts)
2. Replaces the standard MoEGate with ConceptAwareMoEGate
3. Freezes EVERYTHING except the new gate parameters
4. Trains only the gate to learn concept-driven routing
initial part w moe, have concept awarenes, but the aux loss of Diff-MOE reoves it as it wants the tokens to be distributed across all experts , buut we want the tokens to be spread acc to semantic
Key insight: the expert weights are already trained (from Stage 1 finetuning).
We only need the gate to learn WHICH expert to send WHICH concept to.
This is ~50K parameters vs ~50M in full MoE finetuning → trains in hours, not days.

Usage:
    # loss_only mode (recommended first — changes loss, keeps routing architecture)
    python finetune_concept_gate.py \\
        --data-path /local/a/imagenet/imagenet2012/train/ \\
        --ckpt results-finetune/004-DiT-XL-2-MoE-finetune-moe24-27/checkpoints/0115000.pt \\
        --mode loss_only \\
        --epochs 1 --batch-size 16 --lr 3e-4

    # full mode (concept embedding injected into routing decision)
    python finetune_concept_gate.py \\
        --data-path /local/a/imagenet/imagenet2012/train/ \\
        --ckpt results-finetune/004-DiT-XL-2-MoE-finetune-moe24-27/checkpoints/0115000.pt \\
        --mode full \\
        --epochs 1 --batch-size 16 --lr 3e-4
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import shutil
import os

from models import DiT_models, load_pretrained_with_moe
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from moe.concept_gate import ConceptAwareMoEGate


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ]
    )
    return logging.getLogger(__name__)


def replace_gates_with_concept_aware(model, moe_blocks, mode, diversity_weight,
                                       min_expert_util=0.5, collapse_weight=1.0,
                                       gate_temp=1.0):
    """
    Replace the standard MoEGate in each MoE block with ConceptAwareMoEGate.
    Copies over the existing routing weights so we start from the trained state.
    """
    replaced = 0
    for idx in moe_blocks:
        block = model.blocks[idx]
        if not hasattr(block, 'moe') or not hasattr(block.moe, 'gate'):
            continue

        old_gate = block.moe.gate
        new_gate = ConceptAwareMoEGate(
            embed_dim=old_gate.gating_dim,
            num_experts=old_gate.n_routed_experts,
            num_experts_per_tok=old_gate.top_k,
            aux_loss_alpha=old_gate.alpha,
            mode=mode,
            diversity_weight=diversity_weight,
            min_expert_util=min_expert_util,
            collapse_weight=collapse_weight,
            gate_temp=gate_temp,
        )

        # Move new gate to same device as old gate, then copy weights
        device = old_gate.weight.device
        new_gate = new_gate.to(device)
        new_gate.weight.data.copy_(old_gate.weight.data)

        block.moe.gate = new_gate
        replaced += 1

    return replaced


def get_trainable_params(model, moe_blocks, mode):
    """
    Returns only the parameters that should be trained.
    For 'loss_only': just the gate routing weights
    For 'full': gate weights + concept_proj + concept_scale
    """
    trainable = []
    for idx in moe_blocks:
        block = model.blocks[idx]
        if hasattr(block, 'moe') and hasattr(block.moe, 'gate'):
            gate = block.moe.gate
            trainable.append({'params': [gate.weight], 'name': f'block{idx}.gate.weight'})
            if mode == 'full' and hasattr(gate, 'concept_proj'):
                trainable.append({
                    'params': list(gate.concept_proj.parameters()),
                    'name': f'block{idx}.concept_proj'
                })
                trainable.append({
                    'params': [gate.concept_scale],
                    'name': f'block{idx}.concept_scale'
                })
    return trainable


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=3  # LANCZOS
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=3
    )
    arr = transforms.functional.center_crop(pil_image, image_size)
    return arr


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "Training requires a GPU"

    # Setup output directory
    os.makedirs(args.results_dir, exist_ok=True)
    # Find next experiment number
    existing = glob(f"{args.results_dir}/*/")
    exp_num = len(existing)
    exp_name = f"{exp_num:03d}-CDMoE-gate-{args.mode}"
    exp_dir = f"{args.results_dir}/{exp_name}"
    ckpt_dir = f"{exp_dir}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = create_logger(exp_dir)
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Diversity weight: {args.diversity_weight}")

    # ── Build model ──────────────────────────────────────────────────────
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

    # ── Load checkpoint ──────────────────────────────────────────────────
    ckpt_path = args.ckpt
    logger.info(f"Loading checkpoint: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict) and "ema_trainable_only" in raw:
        logger.info(f"Finetuned (EMA) at step={raw.get('train_steps', '?')}")
        backbone_state = find_model("pretrained_models/DiT-XL-2-256x256.pt")
        load_pretrained_with_moe(model, backbone_state)
        model.load_state_dict(raw["ema_trainable_only"], strict=False)
    elif isinstance(raw, dict) and "model_trainable_only" in raw:
        logger.info(f"Finetuned (model) at step={raw.get('train_steps', '?')}")
        backbone_state = find_model("pretrained_models/DiT-XL-2-256x256.pt")
        load_pretrained_with_moe(model, backbone_state)
        model.load_state_dict(raw["model_trainable_only"], strict=False)
    else:
        logger.info("Pretrained checkpoint — loading with load_pretrained_with_moe")
        load_pretrained_with_moe(model, raw)

    # ── Replace gates ────────────────────────────────────────────────────
    n_replaced = replace_gates_with_concept_aware(
        model, moe_blocks, args.mode, args.diversity_weight,
        min_expert_util=args.min_expert_util, collapse_weight=args.collapse_weight,
        gate_temp=args.gate_temp)
    logger.info(f"Replaced {n_replaced} gates with ConceptAwareMoEGate (mode={args.mode})")
    logger.info(f"  diversity_weight={args.diversity_weight}, min_expert_util={args.min_expert_util}, collapse_weight={args.collapse_weight}, gate_temp={args.gate_temp}")

    # ── Freeze everything except gate params ─────────────────────────────
    for param in model.parameters():
        param.requires_grad = False

    trainable_groups = get_trainable_params(model, moe_blocks, args.mode)
    total_trainable = 0
    for group in trainable_groups:
        for p in group['params']:
            p.requires_grad = True
            total_trainable += p.numel()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {total_trainable:,} ({100*total_trainable/total_params:.3f}%)")
    logger.info(f"Trainable groups:")
    for g in trainable_groups:
        n = sum(p.numel() for p in g['params'])
        logger.info(f"  {g['name']}: {n:,} params")

    # ── EMA ───────────────────────────────────────────────────────────────
    # Only track EMA of trainable gate parameters
    ema_state = {}
    for group in trainable_groups:
        for p in group['params']:
            ema_state[id(p)] = p.data.clone()

    def update_ema(decay=0.9999):
        for group in trainable_groups:
            for p in group['params']:
                ema_state[id(p)].mul_(decay).add_(p.data, alpha=1 - decay)

    # ── Optimizer ─────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        [{'params': g['params'], 'lr': args.lr} for g in trainable_groups],
        weight_decay=args.weight_decay,
    )

    # ── Data ──────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # ── VAE ───────────────────────────────────────────────────────────────
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    # ── Diffusion ─────────────────────────────────────────────────────────
    diffusion = create_diffusion(timestep_respacing="")

    # ── Modified forward to pass class labels through ────────────────────
    # We need the gate to receive class labels for the concept-aware loss.
    # The standard forward path: model.forward(x, t, y) computes c = t_embed + y_embed
    # and passes c to each block. The gate inside SparseMoeBlock only gets hidden_states and c.
    #
    # For concept-aware training, we need to thread the raw class labels through.
    # We do this by storing them on the model before each forward call.
    model._current_class_labels = None

    # Monkey-patch the gate forward to include class labels
    original_gate_forwards = {}
    for idx in moe_blocks:
        block = model.blocks[idx]
        if hasattr(block, 'moe') and hasattr(block.moe, 'gate'):
            gate = block.moe.gate
            orig_fwd = gate.forward
            original_gate_forwards[idx] = orig_fwd

            def make_patched_forward(original_forward, parent_model):
                def patched_forward(hidden_states, class_labels=None, class_embeddings=None):
                    # Use stored class labels if not directly provided
                    labels = class_labels if class_labels is not None else parent_model._current_class_labels
                    return original_forward(hidden_states, class_labels=labels, class_embeddings=class_embeddings)
                return patched_forward

            gate.forward = make_patched_forward(orig_fwd, model)

    # ── Training loop ────────────────────────────────────────────────────
    model.train()
    # But keep everything except gates in eval mode
    model.x_embedder.eval()
    model.t_embedder.eval()
    model.y_embedder.eval()
    model.final_layer.eval()
    for block in model.blocks:
        block.norm1.eval()
        block.attn.eval()
        block.norm2.eval()
        if hasattr(block, 'mlp'):
            block.mlp.eval()
        if hasattr(block, 'moe'):
            for expert in block.moe.experts:
                expert.eval()
            if block.moe.n_shared_experts is not None:
                block.moe.shared_experts.eval()

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Starting training...")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Dataset size: {len(dataset):,}")
    logger.info(f"  Steps per epoch: {len(loader):,}")
    if args.max_steps > 0:
        logger.info(f"  Max steps: {args.max_steps} (early stop)")

    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            # Encode to latent space
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            # Store class labels for concept-aware loss
            model._current_class_labels = y

            # Sample timestep and compute diffusion loss
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs=dict(y=y))
            loss = loss_dict["loss"].mean()

            opt.zero_grad()
            loss.backward()

            # Gradient clipping
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in trainable_groups for p in g['params']],
                    args.max_grad_norm
                )

            opt.step()
            update_ema()

            # Logging
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                avg_loss = running_loss / log_steps
                steps_per_sec = log_steps / (time() - start_time)
                gpu_mem = torch.cuda.max_memory_allocated() / 1e9

                # Log gate-specific info + per-expert utilization
                gate_info = ""
                if args.mode == 'full':
                    for idx in moe_blocks[:1]:
                        gate = model.blocks[idx].moe.gate
                        if hasattr(gate, 'concept_scale'):
                            gate_info = f", ConceptScale: {gate.concept_scale.item():.4f}"

                logger.info(
                    f"(step={train_steps:07d}, epoch={epoch}) "
                    f"Train Loss: {avg_loss:.4f}, "
                    f"Steps/Sec: {steps_per_sec:.2f}, "
                    f"GPU Mem: {gpu_mem:.1f}GB{gate_info}"
                )

                # Log per-expert utilization every 500 steps to monitor collapse
                if train_steps % 500 == 0:
                    with torch.no_grad():
                        for idx in moe_blocks:
                            gate = model.blocks[idx].moe.gate
                            # Quick probe: compute routing on last batch's latents
                            test_input = x[:4]  # just 4 samples for speed
                            test_input_embedded = test_input  # already in latent space
                            # Can't easily run gate standalone, so just log weight norms
                            w_norms = gate.weight.norm(dim=1)  # (E,)
                            norm_str = ", ".join(f"E{e}={w_norms[e]:.2f}" for e in range(gate.n_routed_experts))
                            logger.info(f"  Block {idx} gate weight norms: {norm_str}")

                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0:
                # Save only gate state dict
                gate_state = {}
                for idx in moe_blocks:
                    block = model.blocks[idx]
                    if hasattr(block, 'moe') and hasattr(block.moe, 'gate'):
                        gate_state[f'blocks.{idx}.moe.gate'] = {
                            k: v.cpu() for k, v in block.moe.gate.state_dict().items()
                        }

                # Also save EMA gate state
                ema_gate_state = {}
                for group in trainable_groups:
                    for p in group['params']:
                        ema_gate_state[id(p)] = ema_state[id(p)].cpu().clone()

                checkpoint = {
                    'gate_state': gate_state,
                    'ema_gate_state': ema_gate_state,
                    'train_steps': train_steps,
                    'args': args,
                    'mode': args.mode,
                }

                # Save to /tmp first, then move (NFS quota safety)
                tmp_path = f"/tmp/{train_steps:07d}_cdmoe.pt"
                torch.save(checkpoint, tmp_path)
                size_mb = os.path.getsize(tmp_path) / 1e6
                final_path = f"{ckpt_dir}/{train_steps:07d}.pt"
                shutil.move(tmp_path, final_path)
                logger.info(f"Saved checkpoint: {final_path} ({size_mb:.0f} MB)")

            # Early stop for grid search
            if args.max_steps > 0 and train_steps >= args.max_steps:
                logger.info(f"Reached max_steps={args.max_steps}, stopping early.")
                break

        # Break outer epoch loop too
        if args.max_steps > 0 and train_steps >= args.max_steps:
            break

    # ── Save final checkpoint ────────────────────────────────────────────
    gate_state = {}
    for idx in moe_blocks:
        block = model.blocks[idx]
        if hasattr(block, 'moe') and hasattr(block.moe, 'gate'):
            gate_state[f'blocks.{idx}.moe.gate'] = {
                k: v.cpu() for k, v in block.moe.gate.state_dict().items()
            }

    checkpoint = {
        'gate_state': gate_state,
        'train_steps': train_steps,
        'args': args,
        'mode': args.mode,
    }
    tmp_path = "/tmp/final_cdmoe.pt"
    torch.save(checkpoint, tmp_path)
    final_path = f"{ckpt_dir}/final.pt"
    shutil.move(tmp_path, final_path)
    logger.info(f"Saved final checkpoint: {final_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CD-MoE Gate Finetuning")
    # Data
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    # Model
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to Stage 1 finetuned checkpoint")
    parser.add_argument("--moe-blocks", type=int, nargs="+", default=[24, 25, 26, 27])
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--n-shared-experts", type=int, default=2)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--no-dwconv", action="store_true")
    parser.add_argument("--mode", type=str, choices=["standard", "loss_only", "full"],
                        default="loss_only",
                        help="standard=baseline, loss_only=concept loss, full=concept routing+loss")
    parser.add_argument("--diversity-weight", type=float, default=0.5,
                        help="Weight on cross-class diversity loss (v2 default: 0.5)")
    parser.add_argument("--min-expert-util", type=float, default=0.5,
                        help="Anti-collapse: penalize if expert util < this * (1/E)")
    parser.add_argument("--collapse-weight", type=float, default=1.0,
                        help="Weight on anti-collapse penalty")
    parser.add_argument("--gate-temp", type=float, default=1.0,
                        help="Softmax temperature for gate (< 1 = sharper routing)")
    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after this many steps (0=full epoch, for grid search)")
    parser.add_argument("--results-dir", type=str, default="results-cdmoe")

    args = parser.parse_args()
    main(args)
