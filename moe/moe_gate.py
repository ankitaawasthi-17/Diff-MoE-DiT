"""
Diff-MoE reimplementation, Part 1: the Gate (router).

This mirrors kunncheng/Diff-MoE's MoEGate class from models.py.
Comments explain WHY each line exists, not just what it does.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEGate(nn.Module):
    """
    The router. Decides, for every token, which `top_k` experts (out of
    `num_experts`) should process it, and with what weight.

    Key design choice vs. a naive nn.Linear router: the weight matrix is
    a raw nn.Parameter (not wrapped in nn.Linear), initialized with
    kaiming_uniform -- same init PyTorch uses for nn.Linear internally,
    just done explicitly here so it's easy to see/tune.
    """

    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts
        self.alpha = aux_loss_alpha  # weight on the load-balancing loss
        self.norm_topk_prob = False   # whether to renormalize top-k weights to sum to 1
        self.gating_dim = embed_dim

        # (num_experts, embed_dim) -- one row of weights per expert.
        # This is mathematically identical to nn.Linear(embed_dim, num_experts, bias=False),
        # just declared as a raw Parameter for clarity/control.
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        hidden_states: (batch, seq_len, embed_dim) -- e.g. image patch tokens

        Returns:
            topk_idx:    (batch*seq_len, top_k) -- which experts each token picked
            topk_weight: (batch*seq_len, top_k) -- how much weight to give each pick
            aux_loss:    scalar -- load-balancing loss (only computed during training)
        """
        bsz, seq_len, h = hidden_states.shape

        # Flatten batch and sequence dims together -- the router treats
        # every token independently, regardless of which image or which
        # position in the image it came from. This is the "all tokens go
        # through the router in parallel" idea from earlier in our chat.
        hidden_states = hidden_states.reshape(-1, h)  # (batch*seq_len, embed_dim)

        # Raw scores: dot product between each token and each expert's weight row.
        logits = F.linear(hidden_states, self.weight, None)  # (batch*seq_len, num_experts)

        # Softmax -> probabilities that sum to 1 across experts, per token.
        scores = logits.softmax(dim=-1)

        # Keep only the top_k experts per token; everything else is implicitly zero
        # (we just never use it -- we don't even compute it for non-chosen experts).
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # --- Load balancing / auxiliary loss ---
        # Only computed during training, since it's a training signal that
        # discourages expert collapse. Not needed at inference time.
        if self.training and self.alpha > 0.0:
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)  # (batch, seq_len * top_k)

            # one_hot: for every (token, k-th choice), a vector of length
            # num_experts that's 1 at the chosen expert's position, 0 elsewhere.
            mask_ce = F.one_hot(
                topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
            )  # (batch*seq_len*top_k, num_experts)

            # Average across all tokens: what FRACTION of routing decisions
            # actually went to each expert? Call this `ce` (chosen-expert frequency).
            ce = mask_ce.float().mean(0)  # (num_experts,)

            # Average router probability assigned to each expert across all
            # tokens, BEFORE top-k selection. Call this `Pi`.
            Pi = scores.mean(0)  # (num_experts,)

            # fi rescales ce so that "perfectly uniform routing" = 1.0 for every expert.
            fi = ce * self.n_routed_experts

            # The loss: sum over experts of (avg probability) * (rescaled frequency).
            # Intuition: if an expert is BOTH scored highly on average (high Pi)
            # AND chosen disproportionately often (high fi), this product is large
            # and the loss punishes it. If routing is perfectly uniform, fi ≈ 1
            # for every expert and the loss settles near its minimum.
            aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None

        return topk_idx, topk_weight, aux_loss
