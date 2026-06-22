"""
Quick sanity check for DiT_MoE architecture.
Runs on CPU with small tensors — no GPU or pretrained weights needed.
Tests forward/backward in train and eval mode, plus CFG sampling path.
"""
import torch
import sys

def main():
    print("=" * 60)
    print("DiT_MoE Architecture Test")
    print("=" * 60)

    # Use DiT-S/2 config (small, fast) with MoE on blocks 8-11
    from models import DiT_models

    print("\n[1/5] Building DiT_MoE (DiT-S/2 config, 4 MoE blocks)...")
    model = DiT_models['DiT-S/2-MoE'](
        input_size=16,          # small latent size for fast test
        num_classes=10,
        moe_blocks=[8, 9, 10, 11],  # last 4 of 12 blocks
        num_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        rank=32,
        use_dwconv=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Print block types
    for i, block in enumerate(model.blocks):
        block_type = type(block).__name__
        print(f"  Block {i:2d}: {block_type}")

    # Test inputs
    B = 2
    x = torch.randn(B, 4, 16, 16)  # (batch, channels, H, W)
    t = torch.randint(0, 1000, (B,))
    y = torch.randint(0, 10, (B,))

    # [2] Train mode forward + backward
    print("\n[2/5] Forward pass (train mode)...")
    model.train()
    out = model(x, t, y)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    assert out.shape == (B, 8, 16, 16), f"Expected (2, 8, 16, 16), got {out.shape}"
    print("  ✓ Shape correct")

    print("\n[3/5] Backward pass (checking gradient flow)...")
    loss = out.sum()
    loss.backward()

    # Check gradient flow through router
    gate_grads = []
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'moe'):
            gate = block.moe.gate
            if gate.weight.grad is not None:
                grad_norm = gate.weight.grad.norm().item()
                gate_grads.append((i, grad_norm))
                print(f"  Block {i} router grad norm: {grad_norm:.6f}")

    if gate_grads:
        print("  ✓ Gradients flow through MoE router")
    else:
        print("  ✗ WARNING: No gradients in router weights")

    # Check adaLN grads (will be near-zero at init due to zero-init, that's expected)
    for i, block in enumerate(model.blocks):
        adaln_grad = block.adaLN_modulation[-1].weight.grad
        if adaln_grad is not None:
            print(f"  Block {i:2d} adaLN grad norm: {adaln_grad.norm().item():.6f}")
            break  # just check one to save output

    # [4] Eval mode forward
    print("\n[4/5] Forward pass (eval mode / inference path)...")
    model.eval()
    model.zero_grad()
    with torch.no_grad():
        out_eval = model(x, t, y)
    print(f"  Output: {out_eval.shape}")
    assert out_eval.shape == (B, 8, 16, 16), f"Expected (2, 8, 16, 16), got {out_eval.shape}"
    print("  ✓ Shape correct (eval mode)")

    # [5] CFG sampling path
    print("\n[5/5] forward_with_cfg (classifier-free guidance)...")
    # CFG requires doubled batch: first half = conditional, second half = unconditional
    x_cfg = torch.randn(B * 2, 4, 16, 16)
    t_cfg = torch.randint(0, 1000, (B * 2,))
    y_cfg = torch.cat([
        torch.randint(0, 10, (B,)),
        torch.tensor([10] * B)  # null class for CFG
    ])
    with torch.no_grad():
        out_cfg = model.forward_with_cfg(x_cfg, t_cfg, y_cfg, cfg_scale=4.0)
    print(f"  Input:  {x_cfg.shape}")
    print(f"  Output: {out_cfg.shape}")
    assert out_cfg.shape == (B * 2, 8, 16, 16), f"Unexpected shape: {out_cfg.shape}"
    print("  ✓ CFG path works")

    # Check for NaN/Inf
    has_nan = torch.isnan(out_cfg).any().item()
    has_inf = torch.isinf(out_cfg).any().item()
    if has_nan or has_inf:
        print(f"\n  ✗ WARNING: Output contains NaN={has_nan}, Inf={has_inf}")
        sys.exit(1)
    else:
        print("  ✓ No NaN/Inf in output")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
