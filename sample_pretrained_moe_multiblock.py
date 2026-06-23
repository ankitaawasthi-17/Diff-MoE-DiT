import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torchvision.utils import save_image

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from download import find_model
from models import DiT_models

from moe.top2_moe_pretrained import Top2MoEPretrained

import argparse


def main(args):

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    latent_size = args.image_size // 8

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    ckpt_path = (
        args.ckpt
        or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    )

    state_dict = find_model(ckpt_path)

    model.load_state_dict(state_dict)

    if args.mode == "4":

        MOE_BLOCKS = [24, 25, 26, 27]

    elif args.mode == "8":

        MOE_BLOCKS = [
            20, 21, 22, 23,
            24, 25, 26, 27
        ]

    else:

        raise ValueError(
            "mode must be 4 or 8"
        )

    print("\nReplacing blocks:")
    print(MOE_BLOCKS)

    for block_idx in MOE_BLOCKS:

        original_mlp = model.blocks[
            block_idx
        ].mlp

        model.blocks[
            block_idx
        ].mlp = Top2MoEPretrained(
            original_mlp,
            num_experts=4
        ).to(device)

    model.eval()

    diffusion = create_diffusion(
        str(args.num_sampling_steps)
    )

    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{args.vae}"
    ).to(device)

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

    print("\nSampling...\n")

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

    output_file = (
        f"sample_pretrained_moe_{args.mode}blocks.png"
    )

    save_image(
        samples,
        output_file,
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )

    print("\nSUCCESS")
    print(output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="DiT-XL/2"
    )

    parser.add_argument(
        "--vae",
        type=str,
        default="mse"
    )

    parser.add_argument(
        "--image-size",
        type=int,
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

    parser.add_argument(
        "--mode",
        type=str,
        default="4",
        choices=["4", "8"]
    )

    args = parser.parse_args()

    main(args)