import torch
import torch.nn as nn
import numpy as np

# ============================================================
# Predictor
# ============================================================

class SemanticPredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1152, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# Load predictor
# ============================================================

predictor = SemanticPredictor()

predictor.load_state_dict(
    torch.load(
        "results/concept_predictor_generalized.pt"
    )
)

predictor.eval()

# ============================================================
# Load activations
# ============================================================

acts = torch.load(
    "concept_analysis/generated_data/activations.pt"
)

labels = torch.load(
    "concept_analysis/generated_data/labels.pt"
)

N,T,D = acts.shape

# ============================================================
# Predict concepts
# ============================================================

with torch.no_grad():

    logits = predictor(
        acts.reshape(-1, D)
    )

    concepts = logits.argmax(
        dim=-1
    )

concepts = concepts.reshape(
    N,
    T
)

# ============================================================
# Analyze
# ============================================================

unique_labels = sorted(
    labels.unique().tolist()
)

print("\n" + "="*60)
print("CLASS → CONCEPT PERCENTAGES")
print("="*60)

for cls in unique_labels:

    idx = labels == cls

    cls_concepts = concepts[idx]

    cls_concepts = cls_concepts.reshape(-1)

    counts = torch.bincount(
        cls_concepts,
        minlength=5
    ).float()

    pct = (
        counts /
        counts.sum()
    ) * 100

    print(f"\nClass {cls}")

    for c in range(5):

        print(
            f"Concept {c}: "
            f"{pct[c]:.2f}%"
        )

# ============================================================
# Concept Purity
# ============================================================

print("\n" + "="*60)
print("CONCEPT PURITY")
print("="*60)

for concept_id in range(5):

    mask = (
        concepts.reshape(-1)
        == concept_id
    )

    token_labels = (
        labels
        .repeat_interleave(T)
    )[mask]

    counts = torch.bincount(
        torch.tensor([
            unique_labels.index(
                int(x)
            )
            for x in token_labels
        ]),
        minlength=5
    ).float()

    purity = (
        counts.max()
        /
        counts.sum()
    )

    print(
        f"Concept {concept_id}: "
        f"{purity.item()*100:.2f}% purity"
    )

# ============================================================
# Entropy
# ============================================================

print("\n" + "="*60)
print("CONCEPT ENTROPY")
print("="*60)

for concept_id in range(5):

    mask = (
        concepts.reshape(-1)
        == concept_id
    )

    token_labels = (
        labels
        .repeat_interleave(T)
    )[mask]

    counts = torch.bincount(
        torch.tensor([
            unique_labels.index(
                int(x)
            )
            for x in token_labels
        ]),
        minlength=5
    ).float()

    probs = (
        counts /
        counts.sum()
    )

    entropy = -(
        probs *
        torch.log(
            probs + 1e-8
        )
    ).sum()

    print(
        f"Concept {concept_id}: "
        f"{entropy.item():.4f}"
    )