import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

RAW_DIR = Path("../data/raw")
FINAL_DIR = Path("../data/final")

SPECIES = {
    "hibou_grand-duc": "Hibou grand-duc",
    "flamant_rose": "Flamant rose",
    "martin-pecheur": "Martin-pêcheur",
    "cygne_tubercule": "Cygne tuberculé",
    "pic_vert": "Pic vert",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

plt.rcParams["font.family"] = "DejaVu Sans"


def iter_images(folder: Path):
    if not folder.exists():
        return []

    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]


def show_random_grid(species_slug, n=12):
    folder = FINAL_DIR / species_slug
    images = iter_images(folder)

    if not images:
        print("No images found.")
        return

    sample = random.sample(images, min(n, len(images)))

    cols = 4
    rows = (len(sample) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = np.array(axes).flatten()

    fig.suptitle(f"Random images — {SPECIES[species_slug]}", fontsize=14)

    for i, ax in enumerate(axes):
        if i < len(sample):
            img = Image.open(sample[i]).convert("RGB")
            ax.imshow(img)
            ax.set_title(sample[i].name[:15], fontsize=8)

        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_size_distribution():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (slug, label) in enumerate(SPECIES.items()):
        widths = []
        heights = []

        for img_path in iter_images(RAW_DIR / slug):
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception:
                pass

        ax = axes[i]

        ax.hist(widths, bins=20, alpha=0.7, label="Width")
        ax.hist(heights, bins=20, alpha=0.7, label="Height")

        ax.set_title(label)
        ax.set_xlabel("Pixels")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    # hide empty subplot
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Distribution des tailles d'images brutes", fontsize=14)

    plt.tight_layout()
    plt.show()


def show_species_ranking():
    counts = {}

    for slug, label in SPECIES.items():
        counts[label] = len(iter_images(FINAL_DIR / slug))

    labels = sorted(counts, key=counts.get)
    values = [counts[l] for l in labels]

    plt.figure(figsize=(10, 5))

    bars = plt.bar(labels, values)

    plt.title("Nombre d'images par espèce")
    plt.ylabel("Nombre d'images")
    plt.xticks(rotation=15)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 1, str(h), ha="center")

    plt.tight_layout()
    plt.show()

    print("\nEspèces avec le moins d'images :\n")

    for label in labels:
        print(f"{label:<25} : {counts[label]} images")


if __name__ == "__main__":
    show_random_grid("hibou_grand-duc")
    show_size_distribution()
    show_species_ranking()
