import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

DATA_DIR = Path("concept_analysis/generated_data")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(exist_ok=True)

print("Loading activations...")

acts = torch.load(
    DATA_DIR / "activations.pt"
)

N,T,D = acts.shape

print(acts.shape)

tokens = acts.reshape(
    N*T,
    D
)

print(
    "All tokens:",
    tokens.shape
)

# =====================================================
# Sample 20k tokens
# =====================================================

np.random.seed(42)

idx = np.random.choice(
    len(tokens),
    size=20000,
    replace=False
)

tokens = tokens[idx]

print(
    "Sampled:",
    tokens.shape
)

# =====================================================
# PCA
# =====================================================

print("\nRunning PCA...")

pca = PCA(
    n_components=50
)

tokens_pca = pca.fit_transform(
    tokens.numpy()
)

print(
    "Explained variance:",
    pca.explained_variance_ratio_.sum()
)

# =====================================================
# KMeans
# =====================================================

print("\nRunning KMeans...")

kmeans = KMeans(
    n_clusters=8,
    random_state=42,
    n_init=10
)

cluster_ids = kmeans.fit_predict(
    tokens_pca
)

# =====================================================
# Cluster counts
# =====================================================

print("\nCluster sizes:")

with open(
    RESULTS_DIR / "C4_cluster_stats.txt",
    "w"
) as f:

    for c in range(8):

        count = (
            cluster_ids == c
        ).sum()

        pct = (
            100 * count / len(cluster_ids)
        )

        line = (
            f"Cluster {c}: "
            f"{count} "
            f"({pct:.2f}%)"
        )

        print(line)

        f.write(
            line + "\n"
        )

# =====================================================
# Save centers
# =====================================================

torch.save(
    torch.tensor(
        kmeans.cluster_centers_
    ),
    RESULTS_DIR /
    "C4_cluster_centers.pt"
)

# =====================================================
# tSNE
# =====================================================

print("\nRunning TSNE...")

subset = np.random.choice(
    len(tokens_pca),
    size=5000,
    replace=False
)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

emb = tsne.fit_transform(
    tokens_pca[subset]
)

plt.figure(
    figsize=(8,8)
)

scatter = plt.scatter(
    emb[:,0],
    emb[:,1],
    c=cluster_ids[subset],
    s=5
)

plt.title(
    "Token Clusters"
)

plt.savefig(
    RESULTS_DIR /
    "C4_tsne.png",
    dpi=300
)

print(
    "\nSaved:"
)

print(
    RESULTS_DIR /
    "C4_tsne.png"
)

print(
    RESULTS_DIR /
    "C4_cluster_stats.txt"
)

print(
    RESULTS_DIR /
    "C4_cluster_centers.pt"
)