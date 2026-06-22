"""
Diff-MoE reimplementation package.

Provides:
    - MoEGate: the router with load-balancing auxiliary loss
    - MoeMLP_Temporal_Calibration: timestep-conditioned expert MLP
    - MoeMLP_Temporal: shared expert MLP (always active, no routing)
    - SparseMoeBlock_SpatialTemporalMoE: full MoE layer wiring gate + experts
    - DiTBlock_MoE: drop-in replacement for DiTBlock with MoE MLP
"""

from .moe_gate import MoEGate
from .moe_experts import MoeMLP_Temporal_Calibration, MoeMLP_Temporal
from .sparse_moe_block import SparseMoeBlock_SpatialTemporalMoE
from .dit_block_moe import DiTBlock_MoE
