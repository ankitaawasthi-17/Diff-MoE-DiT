"""
Diff-MoE reimplementation, Part 2: the expert.

Mirrors MoeMLP_Temporal_Calibration from the original repo.

This is NOT a plain 2-layer MLP. It's a SwiGLU-style MLP (gate * up,
then down-project) that is ALSO modulated by the timestep/class
conditioning vector `c`, the same adaLN-Zero trick the surrounding DiT
block uses. This is the "time-aware" part of "Diff-MoE: Time-Aware and
Space-Adaptive Experts" -- each expert's behavior shifts depending on
which point in the denoising process it's being used at.

It ALSO multiplies its intermediate activations by `global_info`, a
signal computed once per image (pooled across all tokens) by the
parent block. That's the "globally-aware feature recalibration"
mechanism from the paper -- giving each expert a sense of "what's
this whole image like" in addition to "what's this one token like".
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoeMLP_Temporal_Calibration(nn.Module):
    def __init__(self, hidden_size, intermediate_size, pretraining_tp=1, rank=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.pretraining_tp = pretraining_tp  # tensor-parallel slicing factor; 1 = no slicing

        # Standard SwiGLU components: gate_proj and up_proj both expand
        # hidden_size -> intermediate_size, their outputs are combined
        # (silu(gate) * up), then down_proj projects back down.
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        # LayerNorm with NO learned affine params (elementwise_affine=False)
        # -- this is intentional. The affine part (scale/shift) is instead
        # supplied externally via adaLN_modulation below, conditioned on
        # the timestep. This is the standard adaLN-Zero pattern from DiT.
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Small bottleneck MLP that maps the conditioning vector `c`
        # (timestep + class embedding) down to `rank` dims, then back up
        # to 3 * hidden_size -- producing shift, scale, and gate values.
        # Using a low-rank bottleneck (rank=64) keeps this cheap even
        # though there are many experts, each with their own copy.
        self.rank = rank
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, rank, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rank, 3 * hidden_size, bias=True),
        )
        nn.init.kaiming_uniform_(self.adaLN_modulation[0].weight, a=math.sqrt(5))
        # Zero-init the LAST layer -- standard adaLN-Zero trick. At the
        # start of training, shift=0, scale=0, gate=0, so this expert
        # behaves like an identity function (just passes input through).
        # This makes early training stable: new experts don't immediately
        # destroy the pretrained backbone's behavior.
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, global_info):
        """
        x:           (num_tokens_routed_here, hidden_size) -- only the
                     tokens THIS expert was assigned, already gathered
                     by the parent block.
        c:           (num_tokens_routed_here, hidden_size) -- the
                     timestep+class conditioning vector, broadcast to
                     match each token.
        global_info: (num_tokens_routed_here, intermediate_size) -- a
                     per-image "what's this whole image like" signal,
                     broadcast to match each token, used to rescale
                     this expert's intermediate activations.
        """
        identity = x  # residual connection, applied at the very end

        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale) + shift

        gate_proj = self.gate_proj(x)
        up_proj = self.up_proj(x)

        # The actual "globally-aware feature recalibration": multiply the
        # SwiGLU intermediate activations by a per-image signal, so the
        # expert's output strength can depend on global image content,
        # not just the local token.
        intermediate = self.act_fn(gate_proj) * up_proj * global_info

        down_proj = self.down_proj(intermediate)

        # `gate` (from adaLN_modulation) scales the residual update --
        # at init this is 0, so output == identity (safe starting point).
        return identity + gate * down_proj


class MoeMLP_Temporal(nn.Module):
    """
    Same idea as above, but WITHOUT the global_info recalibration.
    Used for the "shared expert" -- the one expert that processes
    EVERY token (no routing), so it doesn't need the per-image
    recalibration signal in the original design.
    """

    def __init__(self, hidden_size, intermediate_size, pretraining_tp=1, rank=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.rank = rank
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, rank, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rank, 3 * hidden_size, bias=True),
        )
        nn.init.kaiming_uniform_(self.adaLN_modulation[0].weight, a=math.sqrt(5))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        identity = x
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return identity + gate * down_proj
