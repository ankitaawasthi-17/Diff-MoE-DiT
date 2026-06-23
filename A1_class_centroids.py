"""
A1_class_centroids.py
=====================
QUESTION THIS ANSWERS:
  Do different ImageNet classes occupy meaningfully different regions
  in the DiT block-24 activation space?

WHY THIS MATTERS FOR THE PAPER:
  CD-MoE routes tokens to experts based on concept identity.
  If class activations don't cluster, concept routing is pointless.
  If they DO cluster, we have the empirical justification for the entire project.

WHAT TO LOOK FOR IN THE OUTPUT:
  - inter_class_similarity should be LOWER than intra_class_similarity
  - separation_ratio should be > 1.3 (ideally > 2.0)
  - The heatmap should show bright diagonal, dark off-diagonal

INPUTS:  activations.pt  [500, 256, 1152]
         labels.pt        [500]
OUTPUTS: results/A1_centroid_similarity.png
         results/A1_stats.txt
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import torch.nn.functional as F

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path("concept_analysis/generated_data")          # folder containing activations.pt, labels.pt
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {
    207: "golden_retriever",
    360: "otter",
    387: "red_panda",
    417: "balloon",
    279: "arctic_fox",
}

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading activations...")
acts   = torch.load(DATA_DIR / "activations.pt")   # [500, 256, 1152]
labels = torch.load(DATA_DIR / "labels.pt")         # [500]

print(f"  Activations: {acts.shape}")
print(f"  Labels:      {labels.shape}")
print(f"  Classes:     {labels.unique().tolist()}")

# ── Strategy 1: Image-level centroid (mean over all 256 tokens) ──────────────
# Shape: [500, 1152]
image_acts = acts.mean(dim=1)
image_acts_norm = F.normalize(image_acts, dim=-1)

classes = sorted(labels.unique().tolist())
n_classes = len(classes)

# Compute per-class centroid
centroids = {}
for c in classes:
    mask = (labels == c)
    centroids[c] = image_acts_norm[mask].mean(dim=0)  # [1152]
    centroids[c] = F.normalize(centroids[c], dim=0)

# ── Compute pairwise centroid cosine similarity ──────────────────────────────
sim_matrix = torch.zeros(n_classes, n_classes)
for i, ci in enumerate(classes):
    for j, cj in enumerate(classes):
        sim_matrix[i, j] = (centroids[ci] * centroids[cj]).sum().item()

# ── Compute intra vs inter class similarity ──────────────────────────────────
intra_sims = []
inter_sims = []

for i, ci in enumerate(classes):
    mask_i = (labels == ci)
    acts_i  = image_acts_norm[mask_i]   # [100, 1152]

    # Intra-class: pairwise cosine among samples of same class
    sim_ii = (acts_i @ acts_i.T)        # [100, 100]
    # Exclude diagonal
    mask_diag = ~torch.eye(len(acts_i), dtype=torch.bool)
    intra_sims.append(sim_ii[mask_diag].mean().item())

    for j, cj in enumerate(classes):
        if i >= j:
            continue
        mask_j = (labels == cj)
        acts_j  = image_acts_norm[mask_j]
        sim_ij  = (acts_i @ acts_j.T)   # [100, 100]
        inter_sims.append(sim_ij.mean().item())

mean_intra = np.mean(intra_sims)
mean_inter = np.mean(inter_sims)
separation_ratio = mean_intra / (mean_inter + 1e-8)

# ── Strategy 2: Token-level intra/inter class sim ────────────────────────────
# Flatten all tokens: [500*256, 1152]
token_acts = acts.reshape(-1, 1152)
token_labels = labels.unsqueeze(1).expand(-1, 256).reshape(-1)  # [500*256]
token_acts_norm = F.normalize(token_acts, dim=-1)

token_intra = []
token_inter = []

for i, ci in enumerate(classes):
    mask_i = (token_labels == ci)
    # Sample 1000 tokens max for speed
    idx_i = mask_i.nonzero(as_tuple=True)[0]
    idx_i = idx_i[torch.randperm(len(idx_i))[:1000]]
    t_i   = token_acts_norm[idx_i]

    sim_ii = (t_i @ t_i.T)
    mask_d = ~torch.eye(len(t_i), dtype=torch.bool)
    token_intra.append(sim_ii[mask_d].mean().item())

    for j, cj in enumerate(classes):
        if i >= j:
            continue
        mask_j = (token_labels == cj)
        idx_j  = mask_j.nonzero(as_tuple=True)[0]
        idx_j  = idx_j[torch.randperm(len(idx_j))[:1000]]
        t_j    = token_acts_norm[idx_j]
        sim_ij = (t_i @ t_j.T)
        token_inter.append(sim_ij.mean().item())

mean_token_intra = np.mean(token_intra)
mean_token_inter = np.mean(token_inter)
token_sep_ratio  = mean_token_intra / (mean_token_inter + 1e-8)

# ── Print stats ──────────────────────────────────────────────────────────────
stats = f"""
╔══════════════════════════════════════════════════════╗
║           A1: CLASS CENTROID ANALYSIS RESULTS        ║
╠══════════════════════════════════════════════════════╣
║  IMAGE-LEVEL (mean over 256 tokens)                  ║
║  Intra-class cosine similarity : {mean_intra:.4f}           ║
║  Inter-class cosine similarity : {mean_inter:.4f}           ║
║  Separation ratio (intra/inter): {separation_ratio:.4f}           ║
╠══════════════════════════════════════════════════════╣
║  TOKEN-LEVEL (each of 256 patches separately)        ║
║  Intra-class cosine similarity : {mean_token_intra:.4f}           ║
║  Inter-class cosine similarity : {mean_token_inter:.4f}           ║
║  Separation ratio (intra/inter): {token_sep_ratio:.4f}           ║
╠══════════════════════════════════════════════════════╣
║  INTERPRETATION                                      ║"""

if separation_ratio > 2.0:
    verdict = "STRONG separation — concept routing is WELL justified"
elif separation_ratio > 1.3:
    verdict = "MODERATE separation — concept routing is justified"
else:
    verdict = "WEAK separation — rethink approach"

stats += f"\n║  {verdict:<52}║"
stats += f"\n╚══════════════════════════════════════════════════════╝"

print(stats)
with open(OUTPUT_DIR / "A1_stats.txt", "w") as f:
    f.write(stats)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# Panel 1: Centroid similarity heatmap
ax1 = fig.add_subplot(gs[0])
im = ax1.imshow(sim_matrix.numpy(), cmap="RdYlGn", vmin=-0.1, vmax=1.0)
class_labels = [CLASS_NAMES.get(c, str(c)) for c in classes]
ax1.set_xticks(range(n_classes))
ax1.set_yticks(range(n_classes))
ax1.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=9)
ax1.set_yticklabels(class_labels, fontsize=9)
ax1.set_title("Centroid cosine similarity\n(diagonal = same class)", fontsize=10, fontweight="bold")
plt.colorbar(im, ax=ax1, fraction=0.046)
for i in range(n_classes):
    for j in range(n_classes):
        ax1.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center",
                 fontsize=8, color="black" if sim_matrix[i,j] < 0.7 else "white")

# Panel 2: Intra vs Inter bar chart
ax2 = fig.add_subplot(gs[1])
categories = ["Image-level\nIntra", "Image-level\nInter", "Token-level\nIntra", "Token-level\nInter"]
values     = [mean_intra, mean_inter, mean_token_intra, mean_token_inter]
colors     = ["#1D9E75", "#D85A30", "#1D9E75", "#D85A30"]
bars = ax2.bar(categories, values, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
ax2.set_ylabel("Cosine Similarity", fontsize=9)
ax2.set_title("Intra vs Inter-class\nActivation Similarity", fontsize=10, fontweight="bold")
ax2.set_ylim(0, max(values) * 1.25)
for bar, v in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.axhline(y=0, color="gray", linewidth=0.5)
from matplotlib.patches import Patch
ax2.legend(handles=[Patch(color="#1D9E75", label="Intra-class"), Patch(color="#D85A30", label="Inter-class")], fontsize=8)

# Panel 3: Per-class intra-class similarity bar
ax3 = fig.add_subplot(gs[2])
per_class_sims = []
for ci in classes:
    mask_i = (labels == ci)
    acts_i  = image_acts_norm[mask_i]
    sim_ii  = (acts_i @ acts_i.T)
    mask_d  = ~torch.eye(len(acts_i), dtype=torch.bool)
    per_class_sims.append(sim_ii[mask_d].mean().item())
ax3.barh(class_labels, per_class_sims, color="#185FA5", alpha=0.85)
ax3.set_xlabel("Intra-class Cosine Similarity", fontsize=9)
ax3.set_title("Per-class internal\ncohesion", fontsize=10, fontweight="bold")
ax3.axvline(x=mean_inter, color="#D85A30", linestyle="--", linewidth=1.5, label=f"inter-class mean ({mean_inter:.3f})")
ax3.legend(fontsize=8)

fig.suptitle("A1: Class Centroid Analysis — Does concept routing make sense?",
             fontsize=12, fontweight="bold", y=1.02)
plt.savefig(OUTPUT_DIR / "A1_centroid_similarity.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR}/A1_centroid_similarity.png")
print(f"Saved: {OUTPUT_DIR}/A1_stats.txt")