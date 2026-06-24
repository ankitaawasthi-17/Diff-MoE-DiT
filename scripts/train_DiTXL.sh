python finetune_moe.py \\
      --data-path /path/to/imagenet_latents/ \\
      --use-latents \\
      --ckpt DiT-XL-2-256x256.pt \\
      --moe-blocks 20 21 22 23 24 25 26 27 \\
      --num-experts 8 \\
      --epochs 10 \\
      --batch-size 32 \\
      --lr 1e-4