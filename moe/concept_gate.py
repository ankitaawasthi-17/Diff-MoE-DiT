"""
Concept-Aware MoE Gate for CD-MoE (v2).

This replaces the standard MoEGate with one that:
1. Optionally injects class/concept embeddings into the routing decision
2. Uses a concept-aware auxiliary loss that allows concept grouping
   while maintaining load balance WITHIN each concept

The key insight: standard load-balance loss forces globally uniform routing,
which destroys concept specialization. Our loss allows Expert 0 to specialize
on dogs and Expert 1 on geyser, as long as dog-tokens are balanced across
spatial positions within Expert 0.

v2 changes (fixes v1 Block 26 collapse):
- Diversity: replaced entropy-based with pairwise KL divergence (no blind spot)
- Anti-collapse: penalizes any expert falling below min_expert_util * (1/E)
- Default diversity_weight increased from 0.1 to 0.5

Usage in SparseMoeBlock:
    # Replace:  self.gate = MoEGate(...)
    # With:     self.gate = ConceptAwareMoEGate(..., num_classes=1000)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptAwareMoEGate(nn.Module):
    """
    A concept-conditioned router that produces routing decisions informed
    by both token hidden states AND class/concept information.

    Three modes of operation:
    1. mode='standard' — identical to MoEGate (baseline, for comparison)
    2. mode='loss_only' — standard routing, but concept-aware auxiliary loss
    3. mode='full' — concept-conditioned routing AND concept-aware loss

    The concept-aware loss (v2) replaces:
        L_global = sum(Pi * fi)   (penalizes any deviation from uniform)
    With:
        L_within  = sum_c sum(Pi_c * fi_c)       (balance within each concept)
        L_diverse = -mean(KL(class_i || class_j)) (push different classes apart)
        L_collapse = penalty for expert util < min threshold
        L = L_within + λ_div * L_diverse + λ_collapse * L_collapse
    """

    def __init__(
        self,
        embed_dim,
        num_experts=16,
        num_experts_per_tok=2,
        aux_loss_alpha=0.01,
        num_classes=1000,
        concept_embed_dim=256,
        mode='loss_only',           # 'standard', 'loss_only', or 'full'
        diversity_weight=0.5,       # weight on diversity loss term (v2: 0.1→0.5)
        min_expert_util=0.5,        # anti-collapse: penalize if expert < this * (1/E)
        collapse_weight=1.0,        # weight on anti-collapse penalty
        gate_temp=1.0,              # softmax temperature (< 1 = sharper routing)
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts
        self.alpha = aux_loss_alpha
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.mode = mode
        self.diversity_weight = diversity_weight
        self.min_expert_util = min_expert_util
        self.collapse_weight = collapse_weight
        self.gate_temp = gate_temp

        # Standard routing weight (same as MoEGate)
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

        # Concept conditioning (only used in 'full' mode)
        if mode == 'full':
            # Project class embedding into a modulation signal
            self.concept_proj = nn.Sequential(
                nn.Linear(embed_dim, concept_embed_dim),
                nn.SiLU(),
                nn.Linear(concept_embed_dim, self.gating_dim),
            )
            # Learnable scale for how much concept info influences routing
            self.concept_scale = nn.Parameter(torch.tensor(0.1))
            # Initialize last layer to near-zero so concept signal starts small
            nn.init.zeros_(self.concept_proj[-1].weight)
            nn.init.zeros_(self.concept_proj[-1].bias)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states, class_labels=None, class_embeddings=None):
        """
        hidden_states: (batch, seq_len, embed_dim)
        class_labels:  (batch,) integer class labels — needed for concept-aware loss
        class_embeddings: (batch, embed_dim) — the y_embedder output from DiT,
                          needed for 'full' mode concept conditioning

        Returns:
            topk_idx, topk_weight, aux_loss (same interface as MoEGate)
        """
        bsz, seq_len, h = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, h)  # (B*S, D)

        # ── Routing logits ────────────────────────────────────────────────
        if self.mode == 'full' and class_embeddings is not None:
            # Add concept modulation to each token's hidden state before routing
            concept_mod = self.concept_proj(class_embeddings)  # (B, D)
            concept_mod = concept_mod * self.concept_scale
            # Broadcast: (B, 1, D) + (B, S, D)
            modulated = hidden_states + concept_mod.unsqueeze(1)
            logits = F.linear(modulated.reshape(-1, h), self.weight, None)
        else:
            logits = F.linear(hidden_flat, self.weight, None)

        scores = (logits / self.gate_temp).softmax(dim=-1)  # (B*S, E)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # ── Auxiliary loss ────────────────────────────────────────────────
        if self.training and self.alpha > 0.0:
            if self.mode == 'standard' or class_labels is None:
                aux_loss = self._standard_aux_loss(scores, topk_idx, bsz, seq_len)
            else:
                aux_loss = self._concept_aware_aux_loss(
                    scores, topk_idx, class_labels, bsz, seq_len)
        else:
            aux_loss = None

        return topk_idx, topk_weight, aux_loss

    def _standard_aux_loss(self, scores, topk_idx, bsz, seq_len):
        """Original MoEGate aux loss — globally uniform routing."""
        topk_idx_flat = topk_idx.view(bsz, -1)
        mask_ce = F.one_hot(
            topk_idx_flat.view(-1), num_classes=self.n_routed_experts
        )
        ce = mask_ce.float().mean(0)
        Pi = scores.mean(0)
        fi = ce * self.n_routed_experts
        return (Pi * fi).sum() * self.alpha

    def _concept_aware_aux_loss(self, scores, topk_idx, class_labels, bsz, seq_len):
        """
        Concept-aware auxiliary loss v2 with THREE components:

        1. L_within: Load balance WITHIN each concept class.
           Each class's tokens should be evenly distributed among experts.
           But different classes are allowed to prefer different experts.

        2. L_diversity: Pairwise KL divergence between class routing distributions.
           Directly penalizes any two classes having similar expert preferences.
           v2 fix: unlike v1's entropy approach, this has no blind spot when
           all classes collapse to the same expert.

        3. L_collapse: Anti-collapse penalty.
           Penalizes any expert whose global utilization falls below a threshold.
           Prevents the failure mode seen in v1 (Block 26: E0 dropped to 10%).
        """
        E = self.n_routed_experts

        # Reshape scores and topk_idx to (B, S, ...)
        scores_per_image = scores.view(bsz, seq_len, E)      # (B, S, E)
        topk_per_image = topk_idx.view(bsz, seq_len, -1)     # (B, S, top_k)

        # Get unique classes in this batch
        unique_classes = class_labels.unique()

        # ── Component 1: Within-class load balance ────────────────────────
        L_within = torch.tensor(0.0, device=scores.device)
        class_expert_prefs = []  # for diversity computation

        for cls_id in unique_classes:
            # Skip the null class (1000) used in CFG
            if cls_id.item() >= 1000:
                continue

            cls_mask = (class_labels == cls_id)  # (B,) boolean
            if cls_mask.sum() == 0:
                continue

            # Get scores and choices for this class only
            cls_scores = scores_per_image[cls_mask]    # (n_cls, S, E)
            cls_topk = topk_per_image[cls_mask]        # (n_cls, S, top_k)

            # Average probability per expert for this class
            cls_Pi = cls_scores.reshape(-1, E).mean(0)  # (E,)

            # Fraction of routing decisions per expert for this class
            cls_onehot = F.one_hot(
                cls_topk.reshape(-1), num_classes=E
            ).float()  # (n_cls*S*top_k, E)
            cls_ce = cls_onehot.mean(0)  # (E,)
            cls_fi = cls_ce * E

            # Standard load-balance loss, but only for this class's tokens
            L_within = L_within + (cls_Pi * cls_fi).sum()

            # Record this class's expert preference (for diversity)
            # Use cls_Pi (not detached) so diversity gradients flow
            class_expert_prefs.append(cls_Pi)

        # Normalize by number of classes
        n_classes = max(len(class_expert_prefs), 1)
        L_within = L_within / n_classes

        # ── Component 2: Pairwise KL diversity (v2 fix) ───────────────────
        # For every pair of classes, compute symmetric KL divergence between
        # their routing distributions. We MAXIMIZE this (minimize negative).
        # This directly detects and penalizes "all classes → same expert".
        L_diversity = torch.tensor(0.0, device=scores.device)
        if len(class_expert_prefs) >= 2:
            pref_matrix = torch.stack(class_expert_prefs)  # (C, E)
            # Normalize to valid distributions
            pref_matrix = pref_matrix / (pref_matrix.sum(dim=-1, keepdim=True) + 1e-10)

            eps = 1e-10
            C = pref_matrix.shape[0]
            total_kl = torch.tensor(0.0, device=scores.device)
            n_pairs = 0

            for i in range(C):
                for j in range(i + 1, C):
                    p = pref_matrix[i] + eps
                    q = pref_matrix[j] + eps
                    # Symmetric KL: 0.5 * (KL(p||q) + KL(q||p))
                    kl_pq = (p * torch.log(p / q)).sum()
                    kl_qp = (q * torch.log(q / p)).sum()
                    sym_kl = 0.5 * (kl_pq + kl_qp)
                    total_kl = total_kl + sym_kl
                    n_pairs += 1

            if n_pairs > 0:
                avg_kl = total_kl / n_pairs
                # NEGATIVE because we want to MAXIMIZE divergence
                # (minimize -KL = maximize KL)
                L_diversity = -avg_kl

        # ── Component 3: Anti-collapse penalty (v2 new) ───────────────────
        # Compute GLOBAL expert utilization and penalize if any expert
        # falls below min_expert_util * (1/E). This prevents the Block 26
        # failure where E0 dropped to 10% while E3 dominated at 41%.
        L_collapse = torch.tensor(0.0, device=scores.device)
        if self.min_expert_util > 0:
            # Global utilization across all tokens in batch
            global_onehot = F.one_hot(
                topk_idx.view(-1), num_classes=E
            ).float()  # (B*S*top_k, E)
            global_util = global_onehot.mean(0)  # (E,) — fraction per expert

            # Threshold: min_expert_util * uniform = min_expert_util * (1/E)
            threshold = self.min_expert_util / E
            # Penalize experts below threshold using squared hinge loss
            deficit = F.relu(threshold - global_util)  # (E,)
            L_collapse = (deficit ** 2).sum() * E  # Scale by E so loss magnitude is stable

        # ── Combine ───────────────────────────────────────────────────────
        aux_loss = (
            L_within
            + self.diversity_weight * L_diversity
            + self.collapse_weight * L_collapse
        ) * self.alpha

        return aux_loss
