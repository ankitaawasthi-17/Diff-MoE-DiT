import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision.utils import save_image

DATA_DIR = Path(
    "concept_analysis/generated_data"
)

SAVE_DIR = Path(
    "results/token_visualization"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("Loading data...")

images = torch.load(
    DATA_DIR / "images.pt"
)

labels = torch.load(
    DATA_DIR / "labels.pt"
)

print("Images:", images.shape)

# ============================================================
# Top tokens from C1
# ============================================================

TOP_TOKENS = [
    170,
    40,
    96,
    102,
    135
]

# ============================================================
# Token index -> patch coordinates
# ============================================================

def token_to_xy(token_idx):

    row = token_idx // 16
    col = token_idx % 16

    return row, col


# ============================================================
# Draw patch box
# ============================================================

def draw_patch_box(img, token_idx):

    img = img.clone()

    row, col = token_to_xy(
        token_idx
    )

    patch_size = 16

    y0 = row * patch_size
    y1 = y0 + patch_size

    x0 = col * patch_size
    x1 = x0 + patch_size

    # red border

    img[:, y0:y1, x0] = 1
    img[:, y0:y1, x1 - 1] = 1

    img[:, y0, x0:x1] = 1
    img[:, y1 - 1, x0:x1] = 1

    return img


# ============================================================
# Select examples
# ============================================================

examples_per_class = 5

unique_classes = sorted(
    labels.unique().tolist()
)

for cls in unique_classes:

    idxs = torch.where(
        labels == cls
    )[0]

    idxs = idxs[:examples_per_class]

    for token_idx in TOP_TOKENS:

        vis_images = []

        for idx in idxs:

            img = images[idx]

            img = (
                img - img.min()
            ) / (
                img.max() - img.min() + 1e-8
            )

            img = draw_patch_box(
                img,
                token_idx
            )

            vis_images.append(
                img
            )

        vis_images = torch.stack(
            vis_images
        )

        save_path = (
            SAVE_DIR
            /
            f"class_{cls}_token_{token_idx}.png"
        )

        save_image(
            vis_images,
            save_path,
            nrow=5
        )

        print(
            "Saved:",
            save_path
        )

print("\nDone.")