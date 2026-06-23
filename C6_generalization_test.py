import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# Load data
# ============================================================

DATA_DIR = Path(
    "concept_analysis/generated_data"
)

print("Loading data...")

acts = torch.load(
    DATA_DIR / "activations.pt"
)

labels = torch.load(
    DATA_DIR / "labels.pt"
)

print("Acts:", acts.shape)
print("Labels:", labels.shape)

# ============================================================
# Use image-level features
# ============================================================

X = acts.mean(dim=1)

unique_labels = sorted(
    labels.unique().tolist()
)

label_map = {
    lbl: i
    for i, lbl in enumerate(unique_labels)
}

y = torch.tensor(
    [label_map[int(v)] for v in labels]
)

num_classes = len(unique_labels)

print("\nClasses:")
print(unique_labels)

# ============================================================
# Train/Test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X.numpy(),
    y.numpy(),
    test_size=0.20,
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

print("\nTrain samples:", len(X_train))
print("Test samples :", len(X_test))

# ============================================================
# Concept Predictor
# ============================================================

class ConceptPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                1152,
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

    def forward(self, x):

        return self.net(x)

model = ConceptPredictor()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.CrossEntropyLoss()

# ============================================================
# Train
# ============================================================

print("\nTraining...")

for epoch in range(200):

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

    if epoch % 20 == 0:

        with torch.no_grad():

            train_pred = model(
                X_train
            ).argmax(dim=1)

            train_acc = accuracy_score(
                y_train.numpy(),
                train_pred.numpy()
            )

        print(
            f"Epoch {epoch:3d} "
            f"Loss={loss.item():.4f} "
            f"TrainAcc={train_acc*100:.2f}%"
        )

# ============================================================
# Final evaluation
# ============================================================

with torch.no_grad():

    train_pred = model(
        X_train
    ).argmax(dim=1)

    test_pred = model(
        X_test
    ).argmax(dim=1)

train_acc = accuracy_score(
    y_train.numpy(),
    train_pred.numpy()
)

test_acc = accuracy_score(
    y_test.numpy(),
    test_pred.numpy()
)

print("\n===================================")
print("GENERALIZATION RESULTS")
print("===================================")

print(
    f"Train Accuracy : {train_acc*100:.2f}%"
)

print(
    f"Test Accuracy  : {test_acc*100:.2f}%"
)

print("\nConfusion Matrix")

cm = confusion_matrix(
    y_test.numpy(),
    test_pred.numpy()
)

print(cm)

print("\nClassification Report")

print(
    classification_report(
        y_test.numpy(),
        test_pred.numpy(),
        digits=4
    )
)

# ============================================================
# Save model
# ============================================================

torch.save(
    model.state_dict(),
    "results/concept_predictor_generalized.pt"
)

print(
    "\nSaved:"
)

print(
    "results/concept_predictor_generalized.pt"
)