import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = Path(
    "concept_analysis/generated_data"
)

print("Loading activations...")

acts = torch.load(
    DATA_DIR / "activations.pt"
)

labels = torch.load(
    DATA_DIR / "labels.pt"
)

print("Acts:", acts.shape)
print("Labels:", labels.shape)

# ============================================================
# Convert labels to 0..4
# ============================================================

unique_labels = sorted(
    labels.unique().tolist()
)

label_map = {
    lbl: i
    for i, lbl in enumerate(unique_labels)
}

y = torch.tensor(
    [label_map[int(x)] for x in labels]
)

num_classes = len(unique_labels)

print("Classes:", unique_labels)

# ============================================================
# Probe every token position
# ============================================================

token_acc = []

for token_idx in range(256):

    X = acts[:, token_idx, :]

    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(),
        y.numpy(),
        test_size=0.2,
        random_state=42,
        stratify=y.numpy()
    )

    X_train = torch.tensor(
        X_train
    ).float()

    X_test = torch.tensor(
        X_test
    ).float()

    y_train = torch.tensor(
        y_train
    )

    y_test = torch.tensor(
        y_test
    )

    model = nn.Linear(
        1152,
        num_classes
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):

        logits = model(
            X_train
        )

        loss = criterion(
            logits,
            y_train
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    with torch.no_grad():

        preds = model(
            X_test
        )

        preds = preds.argmax(
            dim=1
        )

    acc = accuracy_score(
        y_test.numpy(),
        preds.numpy()
    )

    token_acc.append(acc)

    if token_idx % 32 == 0:

        print(
            f"Token {token_idx:3d} "
            f"Accuracy: {acc:.4f}"
        )

# ============================================================
# Statistics
# ============================================================

token_acc = np.array(
    token_acc
)

print("\n====================================")
print("TOKEN PROBE RESULTS")
print("====================================")

print(
    f"Mean Accuracy : "
    f"{token_acc.mean()*100:.2f}%"
)

print(
    f"Max Accuracy  : "
    f"{token_acc.max()*100:.2f}%"
)

print(
    f"Min Accuracy  : "
    f"{token_acc.min()*100:.2f}%"
)

best_idx = np.argmax(
    token_acc
)

print(
    f"Best Token    : "
    f"{best_idx}"
)

# ============================================================
# Save accuracies
# ============================================================

np.save(
    "results/C1_token_acc.npy",
    token_acc
)

# ============================================================
# Heatmap
# ============================================================

heatmap = token_acc.reshape(
    16,
    16
)

plt.figure(
    figsize=(7,6)
)

plt.imshow(
    heatmap,
    interpolation="nearest"
)

plt.colorbar()

plt.title(
    "Token Concept Accuracy"
)

plt.xlabel(
    "Patch X"
)

plt.ylabel(
    "Patch Y"
)

plt.tight_layout()

plt.savefig(
    "results/C1_token_heatmap.png",
    dpi=300
)

print(
    "\nSaved:"
)

print(
    "results/C1_token_heatmap.png"
)

# ============================================================
# Top 20 tokens
# ============================================================

ranking = np.argsort(
    token_acc
)[::-1]

print(
    "\nTop 20 Tokens:"
)

for i in range(20):

    idx = ranking[i]

    print(
        f"{i+1:2d}. "
        f"Token {idx:3d} "
        f"Acc={token_acc[idx]*100:.2f}%"
    )