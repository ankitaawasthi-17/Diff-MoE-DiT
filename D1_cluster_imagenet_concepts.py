import torch
import numpy as np
import joblib

from pathlib import Path
from sklearn.cluster import MiniBatchKMeans

# =====================================================
# Config
# =====================================================

DATA_DIR = Path(
    "concepts/data"
)

FEATURE_FILE = (
    DATA_DIR /
    "imagenet_block24_features.pt"
)

LABEL_FILE = (
    DATA_DIR /
    "imagenet_concept_labels.pt"
)

KMEANS_FILE = (
    DATA_DIR /
    "kmeans_64.pkl"
)

NUM_CONCEPTS = 64

BATCH_SIZE = 4096

RANDOM_SEED = 42

# =====================================================
# Load Features
# =====================================================

print("\nLoading features...")

features = torch.load(
    FEATURE_FILE
)

print(
    "Feature Shape:",
    features.shape
)

X = features.numpy()

N, D = X.shape

print(
    f"Samples={N}"
)

print(
    f"Dim={D}"
)

# =====================================================
# MiniBatch KMeans
# =====================================================

print(
    f"\nClustering into "
    f"{NUM_CONCEPTS} concepts..."
)

kmeans = MiniBatchKMeans(
    n_clusters=NUM_CONCEPTS,
    batch_size=BATCH_SIZE,
    random_state=RANDOM_SEED,
    verbose=1
)

cluster_ids = kmeans.fit_predict(
    X
)

# =====================================================
# Statistics
# =====================================================

sizes = np.bincount(
    cluster_ids,
    minlength=NUM_CONCEPTS
)

print("\nCluster Sizes")

print(
    "Min:",
    sizes.min()
)

print(
    "Max:",
    sizes.max()
)

print(
    "Mean:",
    sizes.mean()
)

print(
    "Std:",
    sizes.std()
)

# =====================================================
# Save
# =====================================================

torch.save(
    torch.tensor(
        cluster_ids
    ),
    LABEL_FILE
)

joblib.dump(
    kmeans,
    KMEANS_FILE
)

print("\nSaved:")

print(
    LABEL_FILE
)

print(
    KMEANS_FILE
)

print(
    "\nDONE"
)