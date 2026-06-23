import torch
import torch.nn as nn
import numpy as np

print("Loading data...")

acts = torch.load(
    "concept_analysis/generated_data/activations.pt"
)

labels = torch.load(
    "concept_analysis/generated_data/labels.pt"
)

concept_predictor = torch.load(
    "results/concept_predictor.pt",
    weights_only=False
)

N,T,D = acts.shape

print("Acts:", acts.shape)

# ====================================
# STANDARD ROUTER
# ====================================

router = nn.Linear(D, 4)

with torch.no_grad():

    logits = router(
        acts.reshape(-1, D)
    )

    experts = logits.argmax(dim=1)

usage = torch.bincount(
    experts,
    minlength=4
)

print("\n=================================")
print("STANDARD ROUTER")
print("=================================")

print("Usage:", usage.tolist())

# ====================================
# CONCEPT ROUTER
# ====================================

with torch.no_grad():

    concept_logits = concept_predictor(
        acts.reshape(-1, D)
    )

    concepts = concept_logits.argmax(dim=1)

expert_map = {
    0:0,
    1:1,
    2:2,
    3:3,
    4:0
}

concept_experts = torch.tensor(
    [expert_map[int(c)] for c in concepts]
)

usage2 = torch.bincount(
    concept_experts,
    minlength=4
)

print("\n=================================")
print("CONCEPT ROUTER")
print("=================================")

print("Usage:", usage2.tolist())

# ====================================
# SPECIALIZATION
# ====================================

print("\n=================================")
print("CONCEPT DISTRIBUTION PER EXPERT")
print("=================================")

for e in range(4):

    idx = concept_experts == e

    cls = concepts[idx]

    counts = torch.bincount(
        cls,
        minlength=5
    )

    print(f"\nExpert {e}")

    print(
        counts.tolist()
    )

print("\nDone.")