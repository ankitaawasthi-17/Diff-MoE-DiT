import torch
import torch.nn as nn

from pathlib import Path

# ============================================================
# SAME predictor architecture as C6
# ============================================================

class SemanticPredictor(nn.Module):

    def __init__(
        self,
        hidden_size=1152,
        num_classes=5
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                hidden_size,
                256
            ),

            nn.GELU(),

            nn.Linear(
                256,
                64
            ),

            nn.GELU(),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(
        self,
        x
    ):
        return self.net(x)

# ============================================================
# Load predictor
# ============================================================

print("Loading trained predictor...")

predictor = SemanticPredictor()

predictor.load_state_dict(

    torch.load(
        "results/concept_predictor_generalized.pt"
    )

)

predictor.eval()

for p in predictor.parameters():
    p.requires_grad = False

# ============================================================
# Load real activations
# ============================================================

print("Loading activations...")

acts = torch.load(
    "concept_analysis/generated_data/activations.pt"
)

labels = torch.load(
    "concept_analysis/generated_data/labels.pt"
)

print(
    "Acts:",
    acts.shape
)

# ============================================================
# Flatten tokens
# ============================================================

N,T,D = acts.shape

tokens = acts.reshape(
    N*T,
    D
)

print(
    "Tokens:",
    tokens.shape
)

# ============================================================
# Predict concepts
# ============================================================

with torch.no_grad():

    logits = predictor(
        tokens
    )

    probs = torch.softmax(
        logits,
        dim=-1
    )

    concepts = probs.argmax(
        dim=-1
    )

# ============================================================
# Global distribution
# ============================================================

print("\n===================================")
print("GLOBAL CONCEPT DISTRIBUTION")
print("===================================")

counts = torch.bincount(
    concepts,
    minlength=5
)

for i in range(5):

    pct = (
        100 *
        counts[i].item()
        / len(concepts)
    )

    print(
        f"Concept {i}: "
        f"{counts[i].item()} "
        f"({pct:.2f}%)"
    )

# ============================================================
# Per-class concept distribution
# ============================================================

print("\n===================================")
print("CLASS -> CONCEPT")
print("===================================")

concepts = concepts.reshape(
    N,
    T
)

unique_labels = sorted(
    labels.unique().tolist()
)

for cls in unique_labels:

    idx = labels == cls

    cls_concepts = concepts[idx]

    cls_concepts = cls_concepts.reshape(-1)

    counts = torch.bincount(
        cls_concepts,
        minlength=5
    )

    print(
        f"\nClass {cls}"
    )

    print(
        counts.tolist()
    )

print("\nDone.")