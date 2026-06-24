CUDA_VISIBLE_DEVICES=1 python sample_diffmoe.py \
    --ckpt results-finetune/004-DiT-XL-2-MoE-finetune-moe24-27/checkpoints/0105000.pt \
    --moe-blocks 24 25 26 27 \
    --num-experts 4 \
    --num-sampling-steps 100 \
    --seed 42 \
    --output sample_finetuned_105k.png