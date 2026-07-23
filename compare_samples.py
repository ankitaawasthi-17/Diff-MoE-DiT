"""
compare_samples.py — Create side-by-side comparison grids from generated samples.

Usage:
    python compare_samples.py \
        --baseline-dir samples_baseline \
        --cdmoe-dir samples_cdmoe_v2 \
        --output comparison_grid.png
"""

import argparse
import os
from PIL import Image, ImageDraw, ImageFont


# Same 8 test classes
CLASS_NAMES = {
    207: "Golden Retriever", 360: "Otter", 387: "Red Panda",
    974: "Geyser", 88: "Macaw", 979: "Valley", 417: "Balloon", 279: "Arctic Fox",
}


def main(args):
    # Load individual images from both directories
    classes = [207, 360, 387, 974, 88, 979, 417, 279]
    img_size = 256

    baseline_imgs = []
    cdmoe_imgs = []

    for cls in classes:
        name = CLASS_NAMES[cls].replace(' ', '_').lower()

        bpath = os.path.join(args.baseline_dir, f"{name}.png")
        cpath = os.path.join(args.cdmoe_dir, f"{name}.png")

        if os.path.exists(bpath):
            baseline_imgs.append(Image.open(bpath).resize((img_size, img_size)))
        else:
            baseline_imgs.append(Image.new('RGB', (img_size, img_size), (128, 128, 128)))

        if os.path.exists(cpath):
            cdmoe_imgs.append(Image.open(cpath).resize((img_size, img_size)))
        else:
            cdmoe_imgs.append(Image.new('RGB', (img_size, img_size), (128, 128, 128)))

    # Create comparison grid
    n_classes = len(classes)
    label_height = 30
    header_height = 40
    padding = 4
    col_width = img_size + padding
    row_height = img_size + label_height + padding

    total_width = padding + col_width * n_classes
    total_height = header_height + row_height * 2 + padding

    canvas = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        small_font = font

    # Headers
    draw.text((padding + total_width // 4, 10), "Standard Diff-MoE (Baseline)", fill=(0, 0, 0), font=font, anchor="mm")
    draw.text((padding + 3 * total_width // 4, 10), "CD-MoE v2 (Ours)", fill=(0, 100, 200), font=font, anchor="mm")

    # Wait, better layout: rows = models, columns = classes
    # Actually let's do: top row = baseline, bottom row = CD-MoE, with class labels on top

    # Recalculate
    total_width = padding + col_width * n_classes
    total_height = header_height + label_height + row_height * 2 + padding * 2

    canvas = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Class labels at top
    for i, cls in enumerate(classes):
        name = CLASS_NAMES[cls]
        x = padding + i * col_width + img_size // 2
        draw.text((x, 5), name, fill=(0, 0, 0), font=small_font, anchor="mt")

    y_offset = label_height

    # Row labels
    draw.text((2, y_offset + 5), "Baseline", fill=(100, 100, 100), font=small_font)
    draw.text((2, y_offset + row_height + 5), "CD-MoE", fill=(0, 100, 200), font=small_font)

    # Actually, let's make it simpler - just paste images
    for i in range(n_classes):
        x = padding + i * col_width
        # Baseline row
        canvas.paste(baseline_imgs[i], (x, y_offset + 20))
        # CD-MoE row
        canvas.paste(cdmoe_imgs[i], (x, y_offset + 20 + img_size + padding + 20))

    # Add row labels on the left side
    # Re-layout with row label column
    label_col_width = 80
    total_width = label_col_width + col_width * n_classes + padding
    total_height = label_height + 20 + (img_size + padding) * 2 + padding * 4

    canvas = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Class labels at top
    for i, cls in enumerate(classes):
        name = CLASS_NAMES[cls]
        x = label_col_width + i * col_width + img_size // 2
        draw.text((x, 5), name, fill=(0, 0, 0), font=small_font, anchor="mt")

    y1 = label_height + 10  # baseline row y
    y2 = y1 + img_size + padding + 10  # cdmoe row y

    # Row labels
    draw.text((5, y1 + img_size // 2), "Diff-MoE", fill=(100, 100, 100), font=font, anchor="lm")
    draw.text((5, y1 + img_size // 2 + 16), "(baseline)", fill=(150, 150, 150), font=small_font, anchor="lm")
    draw.text((5, y2 + img_size // 2), "CD-MoE", fill=(0, 80, 180), font=font, anchor="lm")
    draw.text((5, y2 + img_size // 2 + 16), "(ours)", fill=(0, 120, 220), font=small_font, anchor="lm")

    # Paste images
    for i in range(n_classes):
        x = label_col_width + i * col_width
        canvas.paste(baseline_imgs[i], (x, y1))
        canvas.paste(cdmoe_imgs[i], (x, y2))

    canvas.save(args.output)
    print(f"Saved comparison: {args.output}")
    print(f"Size: {total_width}x{total_height}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=str, required=True)
    parser.add_argument("--cdmoe-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="comparison_grid.png")
    args = parser.parse_args()
    main(args)
