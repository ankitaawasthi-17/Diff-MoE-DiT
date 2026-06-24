"""
concept_gate.py
===============
Concept-Driven MoE Gate (CD-MoE Router)

The key difference from standard Diff-MoE routing:

  Standard gate:   gate_weight(x)              → expert scores
  CD-MoE gate:     gate_weight(x + proj(c))    → expert scores

Where `c` is a concept/class embedding (e.g., DiT's own label embedding,
or a CLIP embedding). This explicitly conditions routing on WHAT is being
generated, not just the token's hidden state.

Hypothesis: This causes experts to specialize around semantic concepts
(dogs/cats → Expert 0, landscapes → Expert 1, etc.) rather than learning
arbitrary statistical partitions.

Usage:
    gate = ConceptGate(
        hidden_size=1152,
        num_experts=4,
        concept_dim=1152,   # same as DiT's label embedding dim
        top_k=2,
    )
    expert_idx, expert_weights, aux_loss = gate(x, concept_embed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptGate(nn.Module):
    """
    Concept-conditioned sparse router for CD-MoE.

    Routes tokens to experts based on both:
      1. The token's own hidden state (x)
      2. A concept/class embedding (concept_embed)

    The concept embedding is projected into the same space as the token
    hidden state and added before computing expert scores. This makes
    routing semantically aware of what class/concept is being generated.

    Args:
        hidden_size:     Token hidden dimension (e.g. 1152 for DiT-XL/2)
        num_experts:     Number of routed experts
        concept_dim:     Dimension of concept embedding (e.g. 1152 for DiT)
        top_k:           Number of experts selected per token (default: 2)
        aux_loss_coef:   Coefficient for load-balancing auxiliary loss
        concept_scale:   How strongly concept embedding influences routing
                         (1.0 = equal weight to token hidden state)
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        concept_dim: int,
        top_k: int = 2,
        aux_loss_coef: float = 1e-2,
        concept_scale: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.concept_scale = concept_scale

        # Standard gate linear (same as Diff-MoE)
        self.gate_proj = nn.Linear(hidden_size, num_experts, bias=False)

        # Concept projection: maps concept embedding → hidden_size space
        # so it can modulate the gate in the same feature space as tokens
        self.concept_proj = nn.Sequential(
            nn.Linear(concept_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

        # Learnable scale for concept influence (starts at concept_scale,
        # can be learned during finetuning)
        self.concept_scale_param = nn.Parameter(
            torch.ones(1) * concept_scale
        )

        # Initialize gate to near-uniform (standard practice)
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.01)

    def forward(
        self,
        x: torch.Tensor,           # (batch*seq_len, hidden_size)
        concept_embed: torch.Tensor # (batch, concept_dim)  — one per image
    ):
        """
        Args:
            x:             Token hidden states, shape (B*L, D)
            concept_embed: Concept/class embeddings, shape (B, concept_dim)

        Returns:
            topk_idx:     Indices of selected experts, shape (B*L, top_k)
            topk_weight:  Softmax weights for selected experts, shape (B*L, top_k)
            aux_loss:     Load-balancing loss scalar
        """
        BL, D = x.shape

        # Project concept embedding: (B, D_concept) → (B, D_hidden)
        concept_hidden = self.concept_proj(concept_embed)  # (B, D_hidden)

        # Expand concept to match token sequence length
        # We need to figure out seq_len from BL
        # concept_embed is (B, concept_dim), x is (B*L, D)
        B = concept_embed.shape[0]
        L = BL // B

        # concept_hidden: (B, D) → (B, 1, D) → (B, L, D) → (B*L, D)
        concept_expanded = concept_hidden.unsqueeze(1).expand(B, L, D)
        concept_expanded = concept_expanded.reshape(BL, D)

        # Concept-modulated token representation
        x_modulated = x + self.concept_scale_param * concept_expanded

        # Compute expert scores from modulated representation
        logits = self.gate_proj(x_modulated)  # (B*L, num_experts)

        # Top-k selection
        scores = F.softmax(logits, dim=-1)  # (B*L, num_experts)
        topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1)

        # Normalize top-k weights to sum to 1
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        # Load-balancing auxiliary loss (same as standard MoE)
        aux_loss = self._load_balance_loss(scores)

        return topk_idx, topk_weight, aux_loss

    def _load_balance_loss(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Encourages uniform expert utilization across tokens.
        From Switch Transformer (Fedus et al. 2021).
        """
        # scores: (B*L, num_experts)
        # fraction of tokens routed to each expert
        routing_weights = scores.mean(dim=0)  # (num_experts,)
        # ideal: 1/num_experts for each
        # loss penalizes deviation from uniform
        num_tokens = scores.shape[0]
        # compute expert load (hard top-k assignments)
        with torch.no_grad():
            _, topk_idx = torch.topk(scores, self.top_k, dim=-1)
            # one-hot count
            expert_mask = torch.zeros_like(scores)
            expert_mask.scatter_(1, topk_idx, 1.0)
            expert_load = expert_mask.mean(dim=0)  # (num_experts,)

        # aux loss = num_experts * sum(f_i * P_i)
        # where f_i = fraction of tokens routed to expert i (hard)
        # and P_i = mean routing probability to expert i (soft)
        aux_loss = self.num_experts * (expert_load * routing_weights).sum()
        return self.aux_loss_coef * aux_loss


class StandardGate(nn.Module):
    """
    Standard Diff-MoE gate (no concept conditioning) — used as baseline.
    Drop-in replacement for ConceptGate with concept_embed=None.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        aux_loss_coef: float = 1e-2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef

        self.gate_proj = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor, concept_embed=None):
        logits = self.gate_proj(x)
        scores = F.softmax(logits, dim=-1)
        topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        aux_loss = self._load_balance_loss(scores)
        return topk_idx, topk_weight, aux_loss

    def _load_balance_loss(self, scores):
        routing_weights = scores.mean(dim=0)
        with torch.no_grad():
            _, topk_idx = torch.topk(scores, self.top_k, dim=-1)
            expert_mask = torch.zeros_like(scores)
            expert_mask.scatter_(1, topk_idx, 1.0)
            expert_load = expert_mask.mean(dim=0)
        aux_loss = self.num_experts * (expert_load * routing_weights).sum()
        return self.aux_loss_coef * aux_loss
