CUDA_VISIBLE_DEVICES=1 python -m pdb analyze_routing.py \
    --ckpt results-finetune/004-DiT-XL-2-MoE-finetune-moe24-27/checkpoints/0105000.pt \
    --moe-blocks 24 25 26 27 \
    --num-experts 4 \
    --num-sampling-steps 50 \
    --seed 42 \
    --output-dir routing_analysis_105k \
    | tee routing_analysis_105k_summary.txtCUDA_VISIBLE_DEVICES=1 python analyze_routing.py \
    --ckpt results-finetune/004-DiT-XL-2-MoE-finetune-moe24-27/checkpoints/0105000.pt \
    --moe-blocks 24 25 26 27 \
    --num-experts 4 \
    --num-sampling-steps 50 \
    --seed 42 \
    --output-dir routing_analysis_105k \
    | tee routing_analysis_105k_summary.txt