import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

DATA_DIR = Path("concept_analysis/generated_data")

print("Loading data...")

acts = torch.load(DATA_DIR / "activations.pt")
labels = torch.load(DATA_DIR / "labels.pt")

print("Activations:", acts.shape)
print("Labels:", labels.shape)

# ============================================================
# Image-level features
# Average over 256 tokens
# ============================================================

X = acts.mean(dim=1)

print("Feature shape:", X.shape)

# ============================================================
# Convert labels to 0..4
# ============================================================

unique_labels = sorted(labels.unique().tolist())

label_map = {
    lbl: i
    for i, lbl in enumerate(unique_labels)
}

y = torch.tensor(
    [label_map[int(x)] for x in labels]
)

print("Classes:", unique_labels)

# ============================================================
# Train/Test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X.numpy(),
    y.numpy(),
    test_size=0.2,
    random_state=42,
    stratify=y.numpy()
)

# ============================================================
# Linear probe
# ============================================================

model = nn.Linear(
    X.shape[1],
    len(unique_labels)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.CrossEntropyLoss()

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()

y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

print("\nTraining...")

for epoch in range(200):

    logits = model(X_train)

    loss = criterion(
        logits,
        y_train
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch:3d} "
            f"Loss {loss.item():.4f}"
        )

# ============================================================
# Evaluate
# ============================================================

with torch.no_grad():

    preds = model(X_test)

    preds = preds.argmax(dim=1)

acc = accuracy_score(
    y_test.numpy(),
    preds.numpy()
)

print("\n====================================")
print("LINEAR PROBE ACCURACY")
print("====================================")
print(f"{acc*100:.2f}%")

if acc < 0.30:
    print("Very weak concept signal")

elif acc < 0.60:
    print("Moderate concept signal")

else:
    print("Strong concept signal")