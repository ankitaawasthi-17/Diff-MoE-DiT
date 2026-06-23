# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Sample new images from a CD-MoE V2 modified DiT.
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models

from moe.concept_router_moe_v2 import (
    ConceptRouterMoEV2
)

import argparse


def main(args):

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if args.ckpt is None:

        assert (
            args.model == "DiT-XL/2"
        )

        assert (
            args.image_size in [256, 512]
        )

        assert (
            args.num_classes == 1000
        )

    # ============================================================
    # Build DiT
    # ============================================================

    latent_size = (
        args.image_size // 8
    )

    model = DiT_models[
        args.model
    ](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    # ============================================================
    # Load pretrained checkpoint
    # ============================================================

    ckpt_path = (
        args.ckpt
        or
        f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    )

    print("\nLoading pretrained DiT checkpoint...\n")

    state_dict = find_model(
        ckpt_path
    )

    model.load_state_dict(
        state_dict
    )

    print("Checkpoint loaded successfully.")

    # ============================================================
    # Replace Block 24 with CD-MoE V2
    # ============================================================

    print(
        "\nReplacing Block 24 with CD-MoE V2...\n"
    )

    model.blocks[24].mlp = (
        ConceptRouterMoEV2(
            predictor_ckpt=
            "results/concept_predictor_generalized.pt",
            hidden_size=1152,
            mlp_hidden=4608,
            num_experts=4
        ).to(device)
    )

    print(
        model.blocks[24].mlp
    )

    model.eval()

    # ============================================================
    # Diffusion + VAE
    # ============================================================

    diffusion = create_diffusion(
        str(args.num_sampling_steps)
    )

    vae = (
        AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{args.vae}"
        ).to(device)
    )

    # ============================================================
    # Labels
    # ============================================================

    class_labels = [
        207,
        360,
        387,
        974,
        88,
        979,
        417,
        279
    ]

    # ============================================================
    # Noise
    # ============================================================

    n = len(class_labels)

    z = torch.randn(
        n,
        4,
        latent_size,
        latent_size,
        device=device
    )

    y = torch.tensor(
        class_labels,
        device=device
    )

    # ============================================================
    # CFG
    # ============================================================

    z = torch.cat(
        [z, z],
        dim=0
    )

    y_null = torch.tensor(
        [1000] * n,
        device=device
    )

    y = torch.cat(
        [y, y_null],
        dim=0
    )

    model_kwargs = dict(
        y=y,
        cfg_scale=args.cfg_scale
    )

    # ============================================================
    # Sample
    # ============================================================

    print(
        "\nStarting diffusion sampling...\n"
    )

    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device
    )

    samples, _ = samples.chunk(
        2,
        dim=0
    )

    samples = vae.decode(
        samples / 0.18215
    ).sample

    # ============================================================
    # Save
    # ============================================================

    save_image(
        samples,
        "sample_cdmoe_v2.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )

    print(
        "\nSaved: sample_cdmoe_v2.png"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        choices=list(DiT_models.keys()),
        default="DiT-XL/2"
    )

    parser.add_argument(
        "--vae",
        type=str,
        choices=["ema", "mse"],
        default="mse"
    )

    parser.add_argument(
        "--image-size",
        type=int,
        choices=[256, 512],
        default=256
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0
    )

    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=250
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None
    )

    args = parser.parse_args()

    main(args)