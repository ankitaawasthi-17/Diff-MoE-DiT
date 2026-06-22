"""
Diff-MoE reimplementation, Part 4: the full DiT block, integrated.

Mirrors DiTBlock_SpatialTemporalMoE from the original repo, but built
on top of the 3 files we already wrote and tested (moe_gate.py,
moe_experts.py, sparse_moe_block.py) instead of copy-pasting from
the original.

This block replaces the standard "Attention + MLP" DiT block with
"Attention + SparseMoE". Everything else about DiT (patch embedding,
timestep embedding, class embedding, positional embedding, final
output layer) stays the same -- you only swap out individual blocks'
MLP sub-layer.
"""

import torch
import torch.nn as nn

from .sparse_moe_block import SparseMoeBlock_SpatialTemporalMoE


def modulate(x, shift, scale):
    """Standard adaLN-Zero modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock_MoE(nn.Module):
    """
    A DiT block with adaLN-Zero conditioning, where the MLP sub-layer
    is replaced by a Sparse MoE block.

    `attn` is passed in rather than constructed here, so you can plug
    in whatever attention implementation your existing DiT already
    uses (standard, RoPE, flash-attn, etc.) without this file needing
    to know about it.
    """

    def __init__(
        self, hidden_size, num_heads, attn_module,
        mlp_ratio=4, num_experts=8, num_experts_per_tok=2,
        n_shared_experts=2, rank=64, use_dwconv=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attn_module

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.moe = SparseMoeBlock_SpatialTemporalMoE(
            embed_dim=hidden_size, mlp_ratio=mlp_ratio,
            num_experts=num_experts, num_experts_per_tok=num_experts_per_tok,
            n_shared_experts=n_shared_experts, rank=rank,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        # Zero-init so this block starts as close to "pass-through" as
        # possible -- standard adaLN-Zero practice, keeps early training
        # (or finetuning from a pretrained checkpoint) stable.
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # Optional: a depthwise conv mixing spatial neighbors before the
        # MoE layer. This is in the original Diff-MoE repo's block. It's
        # OPTIONAL for your reimplementation -- set use_dwconv=False if
        # you want a simpler block closer to vanilla DiT.
        self.use_dwconv = use_dwconv
        if use_dwconv:
            self.dwconv = nn.Conv2d(
                hidden_size, hidden_size, kernel_size=5, stride=1,
                padding=2, groups=hidden_size, bias=True,
            )

    def forward(self, x, c):
        """
        x: (batch, seq_len, hidden_size) -- image patch tokens
        c: (batch, hidden_size) -- timestep + class conditioning vector
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # --- Attention sub-layer (unchanged from standard DiT) ---
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        # --- MoE sub-layer (replaces the standard MLP) ---
        shortcut = x
        moe_input = modulate(self.norm2(x), shift_mlp, scale_mlp)

        if self.use_dwconv:
            B, seq_len, C = moe_input.shape
            side = int(seq_len ** 0.5)
            assert side * side == seq_len, (
                f"use_dwconv=True requires a square token grid, got seq_len={seq_len}. "
                f"Set use_dwconv=False if your tokens aren't a perfect square grid."
            )
            # (B, seq_len, C) -> (B, C, H, W) for depthwise conv
            moe_input = moe_input.reshape(B, side, side, C).permute(0, 3, 1, 2)
            moe_input = self.dwconv(moe_input)
            # (B, C, H, W) -> (B, seq_len, C)
            moe_input = moe_input.permute(0, 2, 3, 1).reshape(B, -1, C)

        x = shortcut + gate_mlp.unsqueeze(1) * self.moe(moe_input, c)
        return x
