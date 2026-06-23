import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

DATA_DIR = Path("concept_analysis/generated_data")
RESULTS_DIR = Path("results")
SAVE_DIR = RESULTS_DIR / "cluster_patches"

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("Loading data...")

acts = torch.load(
    DATA_DIR / "activations.pt"
)

images = torch.load(
    DATA_DIR / "images.pt"
)

print("Acts:", acts.shape)
print("Images:", images.shape)

N,T,D = acts.shape

# ====================================================
# Flatten tokens
# ====================================================

tokens = acts.reshape(
    N*T,
    D
)

# ====================================================
# PCA
# ====================================================

print("Running PCA...")

pca = PCA(
    n_components=50
)

tokens_pca = pca.fit_transform(
    tokens.numpy()
)

# ====================================================
# Same clustering as C4
# ====================================================

print("Running KMeans...")

kmeans = KMeans(
    n_clusters=8,
    random_state=42,
    n_init=10
)

cluster_ids = kmeans.fit_predict(
    tokens_pca
)

# ====================================================
# Mapping
# ====================================================

image_idx = np.arange(
    N*T
) // T

token_idx = np.arange(
    N*T
) % T

PATCH_SIZE = 16

# ====================================================
# Extract examples
# ====================================================

for cluster in range(8):

    print(
        f"\nCluster {cluster}"
    )

    idxs = np.where(
        cluster_ids == cluster
    )[0]

    np.random.shuffle(
        idxs
    )

    idxs = idxs[:25]

    patches = []

    for idx in idxs:

        img_id = image_idx[idx]
        tok_id = token_idx[idx]

        row = tok_id // 16
        col = tok_id % 16

        y0 = row * PATCH_SIZE
        y1 = y0 + PATCH_SIZE

        x0 = col * PATCH_SIZE
        x1 = x0 + PATCH_SIZE

        img = images[img_id]

        patch = img[
            :,
            y0:y1,
            x0:x1
        ]

        patch = (
            patch - patch.min()
        ) / (
            patch.max()
            - patch.min()
            + 1e-8
        )

        patches.append(
            patch
        )

    patches = torch.stack(
        patches
    )

    grid = torch.zeros(
        3,
        5*PATCH_SIZE,
        5*PATCH_SIZE
    )

    k = 0

    for r in range(5):
        for c in range(5):

            grid[
                :,
                r*PATCH_SIZE:(r+1)*PATCH_SIZE,
                c*PATCH_SIZE:(c+1)*PATCH_SIZE
            ] = patches[k]

            k += 1

    plt.figure(
        figsize=(6,6)
    )

    plt.imshow(
        grid.permute(
            1,2,0
        ).numpy()
    )

    plt.axis("off")

    plt.title(
        f"Cluster {cluster}"
    )

    save_path = (
        SAVE_DIR /
        f"cluster_{cluster}.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        save_path
    )

print("\nDone.")