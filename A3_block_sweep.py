"""
A3_block_sweep.py
=================
QUESTION THIS ANSWERS:
  Which DiT blocks show the strongest concept separation?
  Should we be looking at block 24 (what we collected), or should
  we collect activations at blocks 0, 7, 14, 20, 24, 27?

WHY THIS MATTERS:
  You arbitrarily collected block 24 activations. But your
  Layer IF analysis should inform WHICH blocks to put MoE in.
  This experiment tells you where concept information lives
  in the network — early (low-level), middle, or late (high-level).

WHAT TO RUN:
  This requires generating new activations at multiple blocks.
  It hooks into the DiT model and extracts block-by-block.

WHAT TO LOOK FOR:
  - Separation ratio per block
  - Expect a curve: low at block 0, peaks somewhere in middle/late,
    drops slightly at block 27
  - The peak block(s) = where to put your concept-driven MoE

OUTPUTS: results/A3_block_separation_curve.png
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F
import sys

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {207: "retriever", 360: "otter", 387: "red_panda",
               417: "balloon", 279: "arctic_fox"}
CLASSES = [207, 360, 387, 417, 279]
BLOCKS_TO_PROBE = [0, 4, 8, 12, 16, 20, 22, 24, 26, 27]
N_SAMPLES_PER_CLASS = 20  # small — this is just analysis, not training

def compute_separation(acts_dict, labels):
    """Given {block: [N, 256, 1152]}, compute separation ratio per block."""
    results = {}
    classes = sorted(torch.unique(labels).tolist())

    for block_idx, acts in acts_dict.items():
        # Image-level: mean over tokens
        img_acts = acts.mean(dim=1)  # [N, 1152]
        img_acts = F.normalize(img_acts, dim=-1)

        intra, inter = [], []
        for i, ci in enumerate(classes):
            m_i = (labels == ci)
            a_i = img_acts[m_i]
            if len(a_i) < 2:
                continue
            sim_ii = a_i @ a_i.T
            d = ~torch.eye(len(a_i), dtype=torch.bool)
            intra.append(sim_ii[d].mean().item())
            for cj in classes:
                if cj <= ci:
                    continue
                m_j = (labels == cj)
                a_j = img_acts[m_j]
                inter.append((a_i @ a_j.T).mean().item())

        results[block_idx] = {
            "intra": np.mean(intra) if intra else 0,
            "inter": np.mean(inter) if inter else 0,
            "sep_ratio": np.mean(intra) / (np.mean(inter) + 1e-8) if intra and inter else 0
        }
    return results


def run_with_hooks(model, x, t, y, target_blocks):
    """Run DiT forward pass, capture activations at each target block output."""
    activations = {}

    hooks = []
    for block_idx in target_blocks:
        def make_hook(idx):
            def hook(module, input, output):
                activations[idx] = output.detach().cpu()
            return hook
        h = model.blocks[block_idx].register_forward_hook(make_hook(block_idx))
        hooks.append(h)

    with torch.no_grad():
        _ = model(x, t, y)

    for h in hooks:
        h.remove()

    return activations


# ── Main ─────────────────────────────────────────────────────────────────────
print("Loading DiT model...")
from download import find_model
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DiT_models["DiT-XL/2"](
    input_size=32,
    num_classes=1000
).to(device)
state_dict = find_model(
    "DiT-XL-2-256x256.pt"
)
model.load_state_dict(state_dict)
model.eval()

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
diffusion = create_diffusion(timestep_respacing="")

print(f"Collecting activations at blocks: {BLOCKS_TO_PROBE}")
print(f"  {N_SAMPLES_PER_CLASS} samples × {len(CLASSES)} classes = {N_SAMPLES_PER_CLASS * len(CLASSES)} images")

# Use a single timestep for analysis (t=500 = mid-process)
ANALYSIS_TIMESTEP = 500

all_block_acts = {b: [] for b in BLOCKS_TO_PROBE}
all_labels     = []

for cls in CLASSES:
    print(f"  Generating class {cls} ({CLASS_NAMES[cls]})...")
    y = torch.tensor([cls] * N_SAMPLES_PER_CLASS, device=device)

    # Sample from the model at a fixed timestep
    with torch.no_grad():
        # Start from noise
        z = torch.randn(N_SAMPLES_PER_CLASS, 4, 32, 32, device=device)
        t = torch.full((N_SAMPLES_PER_CLASS,), ANALYSIS_TIMESTEP, device=device, dtype=torch.long)

        # Add noise to match timestep
        noise = torch.randn_like(z)
        alphas_cumprod = torch.tensor(
            diffusion.alphas_cumprod,
            dtype=torch.float32,
            device=device
        )
        sqrt_ac  = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_omc = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        x_t = sqrt_ac * z + sqrt_omc * noise

        # Run forward with hooks
        print("x_t dtype:", x_t.dtype)
        print("model dtype:", next(model.parameters()).dtype)
        block_acts = run_with_hooks(model, x_t, t, y, BLOCKS_TO_PROBE)

    for b in BLOCKS_TO_PROBE:
        if b in block_acts:
            all_block_acts[b].append(block_acts[b])

    all_labels.extend([cls] * N_SAMPLES_PER_CLASS)

# Stack
acts_dict = {}
for b in BLOCKS_TO_PROBE:
    if all_block_acts[b]:
        acts_dict[b] = torch.cat(all_block_acts[b], dim=0)
        print(f"  Block {b:2d}: {acts_dict[b].shape}")

all_labels = torch.tensor(all_labels)

# Compute separation
print("\nComputing separation ratios...")
results = compute_separation(acts_dict, all_labels)

# ── Plot ─────────────────────────────────────────────────────────────────────
blocks       = sorted(results.keys())
sep_ratios   = [results[b]["sep_ratio"] for b in blocks]
intra_vals   = [results[b]["intra"] for b in blocks]
inter_vals   = [results[b]["inter"] for b in blocks]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(blocks, sep_ratios, "o-", color="#534AB7", linewidth=2.5, markersize=8, markerfacecolor="white", markeredgewidth=2)
best_block = blocks[np.argmax(sep_ratios)]
ax.axvline(x=best_block, color="#D85A30", linestyle="--", linewidth=1.5, label=f"Best block: {best_block}")
ax.fill_between(blocks, sep_ratios, alpha=0.12, color="#534AB7")
ax.set_xlabel("DiT Block Index", fontsize=11)
ax.set_ylabel("Separation ratio (intra / inter)", fontsize=11)
ax.set_title("Concept Separation Across Blocks\n(higher = better concept routing candidate)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(blocks, intra_vals, "o-", color="#1D9E75", linewidth=2, markersize=7, label="Intra-class sim")
ax2.plot(blocks, inter_vals, "s--", color="#D85A30", linewidth=2, markersize=7, label="Inter-class sim")
ax2.fill_between(blocks, intra_vals, inter_vals, alpha=0.15, color="#1D9E75")
ax2.set_xlabel("DiT Block Index", fontsize=11)
ax2.set_ylabel("Cosine Similarity", fontsize=11)
ax2.set_title("Intra vs Inter-class Similarity\nPer Block", fontsize=11, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig.suptitle("A3: Block Sweep — Where Does Concept Information Live?",
             fontsize=13, fontweight="bold")
plt.savefig(OUTPUT_DIR / "A3_block_separation_curve.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved: {OUTPUT_DIR}/A3_block_separation_curve.png")
print(f"\nTop-5 blocks by concept separation:")
sorted_blocks = sorted(results.items(), key=lambda x: x[1]["sep_ratio"], reverse=True)
for b, r in sorted_blocks[:5]:
    print(f"  Block {b:2d}: sep_ratio={r['sep_ratio']:.4f}, intra={r['intra']:.4f}, inter={r['inter']:.4f}")

print(f"\n→ These are your MoE target blocks.")