import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = Path(
    "concept_analysis/generated_data"
)

SAVE_DIR = Path(
    "results"
)

SAVE_DIR.mkdir(
    exist_ok=True
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
# Use every token as a training sample
# ============================================================

N, T, D = acts.shape

print(
    f"Images={N} Tokens={T} Dim={D}"
)

# [500,256,1152]
# ->
# [128000,1152]

X = acts.reshape(
    N * T,
    D
)

# Repeat image labels
# [500]
# ->
# [128000]

y = labels.repeat_interleave(
    T
)

unique_labels = sorted(
    labels.unique().tolist()
)

label_map = {
    lbl: i
    for i, lbl in enumerate(unique_labels)
}

y = torch.tensor(
    [label_map[int(v)] for v in y]
)

num_classes = len(unique_labels)

print("Classes:", unique_labels)

# ============================================================
# Train/Test Split
# ============================================================

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

    def forward(
        self,
        x
    ):

        return self.net(x)


model = ConceptPredictor()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.CrossEntropyLoss()

print("\nTraining Concept Predictor...\n")

for epoch in range(30):

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

    if epoch % 5 == 0:

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

        print(
            f"Epoch {epoch:2d} "
            f"Loss={loss.item():.4f} "
            f"Acc={acc*100:.2f}%"
        )

# ============================================================
# Final Accuracy
# ============================================================

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

print("\n=================================")
print("FINAL TEST ACCURACY")
print("=================================")
print(f"{acc*100:.2f}%")

# ============================================================
# Save Predictor
# ============================================================

torch.save(
    model.state_dict(),
    SAVE_DIR / "concept_predictor.pt"
)

print(
    "\nSaved:"
)

print(
    SAVE_DIR / "concept_predictor.pt"
)