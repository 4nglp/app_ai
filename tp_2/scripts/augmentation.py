import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SOURCE_DIR = Path("../data/cleaned")
AUG_DIR = Path("../data/final")
LOG_DIR = Path("../logs")

SPECIES = [
    "hibou_grand-duc",
    "flamant_rose",
    "martin-pecheur",
    "cygne_tubercule",
    "pic_vert",
]

AUGMENTATIONS_PER_IMAGE = 5
TARGET_SIZE = (224, 224)

# augmentation ranges
ROTATION_RANGE = (-15, 15)  # degrees
ZOOM_RANGE = (0.9, 1.1)  # scale factor
BRIGHTNESS_RANGE = (0.75, 1.25)  # multiplier
CONTRAST_RANGE = (0.75, 1.25)  # multiplier

VIS_SAMPLES = 3


LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"augmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def augment_image(img: Image.Image) -> Image.Image:
    result = img.copy()

    # flip horizontal 50% chance
    if random.random() < 0.5:
        result = ImageOps.mirror(result)

    # rotation
    angle = random.uniform(*ROTATION_RANGE)
    result = result.rotate(angle, resample=Image.BICUBIC, expand=False)

    # zoom (crop + resize back)
    zoom = random.uniform(*ZOOM_RANGE)
    w, h = result.size
    if zoom < 1.0:
        # zoom out: add black padding then resize back
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        small = result.resize((new_w, new_h), Image.LANCZOS)
        padded = Image.new("RGB", (w, h), (0, 0, 0))
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2
        padded.paste(small, (offset_x, offset_y))
        result = padded
    else:
        # zoom in: crop center then resize back
        crop_w = int(w / zoom)
        crop_h = int(h / zoom)
        x0 = (w - crop_w) // 2
        y0 = (h - crop_h) // 2
        result = result.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        result = result.resize((w, h), Image.LANCZOS)

    # brightness
    brightness_factor = random.uniform(*BRIGHTNESS_RANGE)
    result = ImageEnhance.Brightness(result).enhance(brightness_factor)

    # contrast
    contrast_factor = random.uniform(*CONTRAST_RANGE)
    result = ImageEnhance.Contrast(result).enhance(contrast_factor)

    return result


def augment_species(species: str) -> int:
    src_folder = SOURCE_DIR / species
    dst_folder = AUG_DIR / species

    if not src_folder.exists():
        logger.warning(f"Source folder not found: {src_folder} — skipped.")
        return 0

    dst_folder.mkdir(parents=True, exist_ok=True)

    images = sorted(
        [
            p
            for p in src_folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )

    if not images:
        logger.warning(f"No images in {src_folder}")
        return 0

    logger.info(
        f"Augmenting : {species}  ({len(images)} originals × {AUGMENTATIONS_PER_IMAGE})"
    )

    total_generated = 0

    for img_path in images:
        try:
            with Image.open(img_path) as orig:
                orig = orig.convert("RGB")
        except Exception as e:
            logger.warning(f"Cannot open {img_path.name}: {e}")
            continue

        for i in range(1, AUGMENTATIONS_PER_IMAGE + 1):
            aug = augment_image(orig)
            out_name = f"{img_path.stem}_aug{i:02d}.jpg"
            out_path = dst_folder / out_name
            try:
                aug.save(out_path, format="JPEG", quality=92, optimize=True)
                total_generated += 1
            except Exception as e:
                logger.warning(f"Save failed for {out_name}: {e}")

        logger.info(f"  {img_path.name} → {AUGMENTATIONS_PER_IMAGE} augmented images")

    logger.info(f"Total generated for '{species}': {total_generated} images")
    return total_generated


def visualise_augmentations(species: str, n_originals: int = VIS_SAMPLES):
    src_folder = SOURCE_DIR / species
    dst_folder = AUG_DIR / species

    originals = sorted(
        [
            p
            for p in src_folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )

    if not originals:
        logger.warning(f"No originals for visualisation: {species}")
        return

    sample = random.sample(originals, min(n_originals, len(originals)))

    cols = 1 + AUGMENTATIONS_PER_IMAGE  # original + N augmented
    rows = len(sample)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle(f"Augmentation {species}", fontsize=13, y=1.01)

    # make axes always 2D for uniform indexing
    if rows == 1:
        axes = [axes]

    for row, orig_path in enumerate(sample):
        ax_row = axes[row]

        # column 0: original
        try:
            with Image.open(orig_path) as img:
                ax_row[0].imshow(img.convert("RGB"))
        except Exception:
            ax_row[0].text(0.5, 0.5, "Error", ha="center", va="center")
        ax_row[0].set_title("Original", fontsize=8)
        ax_row[0].axis("off")

        # columns 1…N: augmented versions
        for i in range(1, AUGMENTATIONS_PER_IMAGE + 1):
            aug_name = f"{orig_path.stem}_aug{i:02d}.jpg"
            aug_path = dst_folder / aug_name
            try:
                with Image.open(aug_path) as aug_img:
                    ax_row[i].imshow(aug_img.convert("RGB"))
            except Exception:
                ax_row[i].text(0.5, 0.5, "N/A", ha="center", va="center")
            ax_row[i].set_title(f"Aug {i}", fontsize=8)
            ax_row[i].axis("off")

    plt.tight_layout()
    out_png = AUG_DIR / f"augmentation_{species}.png"
    plt.savefig(out_png, bbox_inches="tight", dpi=100)
    plt.show()
    logger.info(f"Visualisation saved on {out_png}")


def main():
    logger.info("Data Augmentation")
    logger.info(f"Source : {SOURCE_DIR}")
    logger.info(f"Output : {AUG_DIR}")
    logger.info(f"Log    : {log_file}\n")

    AUG_DIR.mkdir(parents=True, exist_ok=True)

    grand_total = 0
    for species in SPECIES:
        grand_total += augment_species(species)

    logger.info(f"Total images generated : {grand_total}")

    logger.info("Generating visualisation grids…")
    for species in SPECIES:
        visualise_augmentations(species)

    logger.info("Augmentation complete.")


if __name__ == "__main__":
    main()
