"""
Diff-MoE reimplementation, Part 3: the full Sparse MoE block.

Mirrors SparseMoeBlock_SpatialTemporalMoE from the original repo.

This wires together:
    - MoEGate          (moe_gate.py)       -- decides routing
    - MoeMLP_Temporal_Calibration (moe_experts.py) -- the routed experts
    - MoeMLP_Temporal  (moe_experts.py)    -- the always-on "shared expert"
    - a small squeeze-excite-style network -- produces `global_info`

It has TWO different code paths for actually running the experts:
    - forward() training path: every token is duplicated `top_k` times
      (e.g. 2x) and run through whichever expert each duplicate was
      assigned to. This is simple and fully vectorizable, which matters
      for backprop, but uses 2x the compute of a "true" sparse pass.
    - moe_infer(): an inference-only path that sorts tokens by which
      expert they go to, and processes each expert's tokens as one
      batched matmul -- no duplication, true sparse compute. This only
      runs under @torch.no_grad() since it doesn't need to support
      backprop.

This split (denser-but-simple for training, truly-sparse for inference)
is a common and intentional design pattern in MoE implementations, not
a hack.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_gate import MoEGate
from .moe_experts import MoeMLP_Temporal_Calibration, MoeMLP_Temporal


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    A trick to make the scalar `aux_loss` contribute to backprop even
    though it doesn't flow through the normal output `y`.

    forward(): returns `x` unchanged -- aux_loss has NO effect on the
               forward numerical result.
    backward(): injects a gradient of 1.0 for `aux_loss`'s contribution,
                so calling .backward() on (main_loss + this) correctly
                updates the router's weights to reduce aux_loss too.

    Why bother with this instead of just writing
    `total_loss = main_loss + aux_loss` directly? Because in the real
    repo, `y` (not the loss) is what gets passed around and combined
    with other tensors deep inside the model before a final loss is
    computed far away. This trick lets the aux_loss "hitch a ride" on
    `y` so it doesn't get lost, without changing y's actual values.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class SparseMoeBlock_SpatialTemporalMoE(nn.Module):
    def __init__(
        self, embed_dim, mlp_ratio=4, num_experts=16, num_experts_per_tok=2,
        pretraining_tp=1, n_shared_experts=2, rank=64,
    ):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok

        # The N routed experts -- each token picks num_experts_per_tok of these.
        self.experts = nn.ModuleList([
            MoeMLP_Temporal_Calibration(
                hidden_size=embed_dim,
                intermediate_size=int(mlp_ratio * embed_dim),
                pretraining_tp=pretraining_tp, rank=rank,
            )
            for _ in range(num_experts)
        ])

        self.gate = MoEGate(embed_dim=embed_dim, num_experts=num_experts,
                             num_experts_per_tok=num_experts_per_tok)

        # --- "Globally-aware feature recalibration" signal ---
        # A tiny squeeze-excite style network: pool all tokens in the
        # image down to one vector, project through a bottleneck, then
        # back up to intermediate_size with a Sigmoid -- producing a
        # per-channel "how relevant is this channel for this image"
        # signal in range (0, 1). Every routed expert multiplies its
        # SwiGLU intermediate activations by this.
        self.se = nn.Sequential(
            nn.Linear(embed_dim, rank, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rank, int(mlp_ratio * embed_dim), bias=False),
            nn.Sigmoid(),
        )

        # --- Shared expert(s) ---
        # Always active for every token, no routing. Acts as a stable
        # "default pathway" alongside the more volatile routed experts.
        # This is the same idea used in DeepSeek-MoE.
        self.n_shared_experts = n_shared_experts
        if self.n_shared_experts is not None:
            intermediate_size = embed_dim * self.n_shared_experts
            self.shared_experts = MoeMLP_Temporal(
                hidden_size=embed_dim, intermediate_size=intermediate_size,
                pretraining_tp=pretraining_tp, rank=rank,
            )

    def forward(self, hidden_states, c):
        """
        hidden_states: (batch, seq_len, embed_dim) -- image patch tokens
        c:             (batch, embed_dim) -- timestep+class conditioning,
                       ONE vector per image (not per token)
        """
        identity = hidden_states
        orig_shape = hidden_states.shape  # (batch, seq_len, embed_dim)

        # --- Compute the global recalibration signal ---
        # Pool across the seq_len dimension -> one vector per image.
        pooled_token = F.adaptive_avg_pool2d(
            hidden_states, output_size=(1, hidden_states.size(2))
        ).squeeze(1)  # (batch, embed_dim)
        global_info = self.se(pooled_token)  # (batch, intermediate_size)

        # --- Routing decision (same for every token, computed once) ---
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        # topk_idx, topk_weight: (batch*seq_len, top_k)

        # Flatten tokens across batch and sequence -- router treats all
        # tokens independently regardless of which image they're from.
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])  # (batch*seq_len, embed_dim)

        # `c` and `global_info` are currently one-per-image. Repeat each
        # image's vector once per token in that image, so shapes line up.
        c_per_token = c.repeat_interleave(int(orig_shape[1]), dim=0)            # (batch*seq_len, embed_dim)
        global_info_per_token = global_info.repeat_interleave(int(orig_shape[1]), dim=0)  # (batch*seq_len, intermediate_size)

        flat_topk_idx = topk_idx.view(-1)  # (batch*seq_len*top_k,)

        if self.training:
            y = self._forward_train(
                hidden_states, c_per_token, global_info_per_token,
                flat_topk_idx, topk_weight, orig_shape,
            )
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(
                hidden_states, c_per_token, flat_topk_idx,
                topk_weight.view(-1, 1), global_info_per_token,
            ).view(*orig_shape)

        if self.n_shared_experts is not None:
            # Shared expert sees the ORIGINAL (un-flattened) tokens,
            # shape (batch, seq_len, embed_dim). Its adaLN_modulation
            # expects a conditioning vector of shape (..., embed_dim)
            # that broadcasts against `identity` -- so we need
            # (batch, 1, embed_dim), NOT (batch, seq_len, embed_dim).
            # unsqueeze(1) broadcasts correctly across the seq_len dim
            # via PyTorch's normal broadcasting rules.
            c_broadcast = c.unsqueeze(1)  # (batch, 1, embed_dim)
            y = y + self.shared_experts(identity, c_broadcast)

        return y

    def _forward_train(self, hidden_states, c_per_token, global_info_per_token,
                        flat_topk_idx, topk_weight, orig_shape):
        """
        Training path: duplicate every token `top_k` times (once per
        chosen expert), run each duplicate through its assigned expert,
        then combine the `top_k` outputs per token using the router's
        weights.

        This is simple and fully differentiable, at the cost of doing
        top_k times the expert compute vs. a true sparse pass. That
        tradeoff is worth it during training because we need gradients
        to flow cleanly through everything.
        """
        # Duplicate each token top_k times (e.g. 2x), matching flat_topk_idx's length.
        hidden_states_rep = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
        c_rep = c_per_token.repeat_interleave(self.num_experts_per_tok, dim=0)
        global_info_rep = global_info_per_token.repeat_interleave(self.num_experts_per_tok, dim=0)

        y = torch.empty_like(hidden_states_rep, dtype=hidden_states_rep.dtype)

        for i, expert in enumerate(self.experts):
            mask = flat_topk_idx == i
            if mask.sum() == 0:
                continue
            y[mask] = expert(
                hidden_states_rep[mask], c_rep[mask], global_info_rep[mask]
            ).to(y.dtype)

        # Reshape back to (batch*seq_len, top_k, embed_dim), weight each
        # of the top_k outputs by its router weight, and sum them.
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        y = y.view(*orig_shape)
        return y

    @torch.no_grad()
    def moe_infer(self, x, c, flat_expert_indices, flat_expert_weights, global_info):
        """
        Inference path: TRUE sparse compute, no duplication.

        Strategy: sort all (token, chosen-expert) pairs by expert index.
        This groups all tokens going to expert 0 together, then expert 1,
        etc. Then we can process each expert's tokens as ONE batched
        matmul (the expensive part of an MLP), instead of doing a python
        loop with per-token overhead.

        `flat_expert_indices` has length batch*seq_len*top_k (each token
        appears top_k times, once per chosen expert).
        """
        expert_cache = torch.zeros_like(x)

        # argsort groups same-expert entries together.
        idxs = flat_expert_indices.argsort()

        # bincount: how many (token, choice) pairs go to each expert.
        # cumsum: running total, so tokens_per_expert[i] tells us where
        # expert i's block of sorted tokens ENDS in `idxs`.
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

        # Each entry in flat_expert_indices corresponds to one
        # (token, k-th choice) pair. Integer-divide by num_experts_per_tok
        # to recover which original token each sorted entry came from.
        token_idxs = idxs // self.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue  # this expert got zero tokens this batch

            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]

            expert_tokens = x[exp_token_idx]
            expert_c = c[exp_token_idx]
            expert_global = global_info[exp_token_idx]

            expert_out = expert(expert_tokens, expert_c, expert_global)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            expert_cache = expert_cache.to(expert_out.dtype)
            # scatter_reduce_ with reduce='sum': adds expert_out's rows
            # into expert_cache at the original token positions. Since
            # each token's top_k contributions get scattered in across
            # different iterations of this loop, 'sum' correctly
            # accumulates both (or all top_k) contributions per token.
            expert_cache.scatter_reduce_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out, reduce='sum',
            )

        return expert_cache
