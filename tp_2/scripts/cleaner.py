import os
import shutil
import random
import logging
from pathlib import Path
from datetime import datetime

import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = Path("../data/raw")
LOG_DIR = Path("../logs")
CLEAN_DIR = Path("../data/cleaned")
WATERMARK_DIR = Path("../data/watermarks")
MIN_SIZE = 200
HASH_SIZE = 8  # bits per side for perceptual hash
HASH_THRESHOLD = 5  # hamming distance ≤ this duplicate
SAMPLE_PER_CLASS = 10  # images for manual review grid

WATERMARK_EDGE_RATIO_THRESHOLD = (
    0.15  # fraction of edge pixels then it's likely a watermark
)
WATERMARK_LOW_VAR_THRESHOLD = 200.0  # variance below this in a region possible logo

SPECIES = [
    "hibou_grand-duc",
    "flamant_rose",
    "martin-pecheur",
    "cygne_tubercule",
    "pic_vert",
]


LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"cleaning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def iter_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    return [p for p in folder.iterdir() if p.suffix.lower() in exts]


def remove_duplicates(folder: Path) -> int:
    # uses perceptual hashing (pHash) to detect near-duplicate images.
    # keeps the first occurrence and removes the rest.
    # returns the number of duplicates removed.
    logger.info(f"[Duplicates] analysis of {folder.name}…")
    images = iter_images(folder)
    hashes: dict[str, Path] = {}  # hash_str to first path seen
    removed = 0

    for img_path in images:
        try:
            with Image.open(img_path) as img:
                h = imagehash.phash(img, hash_size=HASH_SIZE)
        except Exception as e:
            logger.warning(f"Hashing impossible of {img_path.name}: {e}")
            continue

        # compare against all stored hashes
        is_dup = False
        for stored_hash_str, stored_path in hashes.items():
            stored_hash = imagehash.hex_to_hash(stored_hash_str)
            if abs(h - stored_hash) <= HASH_THRESHOLD:
                logger.info(f"Duplicate: {img_path.name} ≈ {stored_path.name} deleted")
                img_path.unlink()
                removed += 1
                is_dup = True
                break

        if not is_dup:
            hashes[str(h)] = img_path

    logger.info(f"[Duplicates] {removed} duplicates deleted on {folder.name}")
    return removed


def filter_by_size(folder: Path, min_px: int = MIN_SIZE) -> int:
    logger.info(f"[Size] Filter ≥ {min_px}px on {folder.name}…")
    images = iter_images(folder)
    removed = 0

    for img_path in images:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            continue

        if w < min_px or h < min_px:
            logger.info(f"A lil small ({w}×{h}): {img_path.name} deleted")
            img_path.unlink()
            removed += 1

    logger.info(f"[Size] {removed} small sized images deleted")
    return removed


def remove_corrupted(folder: Path) -> int:
    logger.info(f"[Corruption] Verification on {folder.name}…")
    images = iter_images(folder)
    removed = 0

    for img_path in images:
        try:
            with Image.open(img_path) as img:
                img.verify()  # Checks file integrity
            # verify() closes the file; re-open to actually decode pixels
            with Image.open(img_path) as img:
                img.load()
        except (UnidentifiedImageError, OSError, SyntaxError, Exception) as e:
            logger.warning(f"Corrupted: {img_path.name} ({e}) deleted")
            img_path.unlink()
            removed += 1

    logger.info(f"[Corruption] {removed} corrupted images deleted")
    return removed


def has_watermark(img_path: Path) -> bool:
    try:
        with Image.open(img_path).convert("L") as gray:
            arr = np.array(gray, dtype=np.float32)
    except Exception:
        return False

    h, w = arr.shape

    # bottom right
    corner = arr[h // 2 :, w // 2 :]
    # sobel like
    gx = np.abs(np.diff(corner, axis=1))
    gy = np.abs(np.diff(corner, axis=0))
    edge_density = (gx > 20).mean() + (gy > 20).mean()
    if edge_density > WATERMARK_EDGE_RATIO_THRESHOLD * 2:
        return True

    # text overlay
    band_height = max(1, h // 8)
    for start in range(0, h - band_height, band_height):
        band = arr[start : start + band_height, :]
        if band.var() < WATERMARK_LOW_VAR_THRESHOLD and band.mean() > 220:
            return True

    return False


def flag_watermarks(folder: Path) -> list[Path]:
    logger.info(f"[Filigrane] detection on {folder.name}…")

    species_watermark_dir = WATERMARK_DIR / folder.name
    species_watermark_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(folder)
    flagged = []

    for img_path in images:
        if has_watermark(img_path):
            dest = species_watermark_dir / img_path.name

            shutil.move(str(img_path), dest)

            logger.info(f"Watermark suspected: {img_path.name} moved to {dest}")

            flagged.append(dest)

    logger.info(f"[Filigrane] {len(flagged)} images moved to {species_watermark_dir}")

    return flagged


# def flag_watermarks(folder: Path) -> list[Path]:
#     logger.info(f"[Filigrane] detection on {folder.name}…")

#     species_watermark_dir = WATERMARK_DIR / folder.name
#     species_watermark_dir.mkdir(parents=True, exist_ok=True)

#     images = iter_images(folder)
#     flagged = []

#     for img_path in images:
#         if has_watermark(img_path):
#             dest = species_watermark_dir / img_path.name

#             shutil.copy2(img_path, dest)

#             logger.info(f"Watermark suspected: {img_path.name} copied to {dest}")

#             flagged.append(dest)

#     logger.info(
#         f"[Filigrane] {len(flagged)} watermark images copied to {species_watermark_dir}"
#     )

#     return flagged


def manual_review_grid(folder: Path, n: int = SAMPLE_PER_CLASS):
    images = iter_images(folder)
    if not images:
        logger.warning(f"[Revue] No image on {folder.name}")
        return

    sample = random.sample(images, min(n, len(images)))
    cols = 5
    rows = (len(sample) + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 3, rows * 3))
    fig.suptitle(
        f"Revue manuelle – {folder.name} (échantillon aléatoire)", fontsize=14, y=1.01
    )
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)

    for i, img_path in enumerate(sample):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        try:
            with Image.open(img_path) as img:
                ax.imshow(img.convert("RGB"))
        except Exception:
            ax.text(0.5, 0.5, "Erreur", ha="center", va="center")
        ax.set_title(img_path.name[:20], fontsize=7)
        ax.axis("off")

    # hides unused axes
    for j in range(len(sample), rows * cols):
        fig.add_subplot(gs[j // cols, j % cols]).axis("off")

    out_png = LOG_DIR / f"review_{folder.name}.png"
    plt.savefig(out_png, bbox_inches="tight", dpi=100)
    plt.show()
    logger.info(f"[Revue] Grille sauvegardée {out_png}")


def clean_species(species_slug: str):
    raw_folder = BASE_DIR / species_slug

    if not raw_folder.exists():
        logger.warning(f"Couldn't found folder: {raw_folder} ignored.")
        return

    cleaned_folder = CLEAN_DIR / species_slug

    if cleaned_folder.exists():
        shutil.rmtree(cleaned_folder)

    cleaned_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Copying raw images for {species_slug}...")

    for img_path in iter_images(raw_folder):
        shutil.copy2(img_path, cleaned_folder / img_path.name)

    logger.info(f"Cleaning: {species_slug}")

    before = len(iter_images(cleaned_folder))
    logger.info(f"Images before cleaning: {before}")

    remove_corrupted(cleaned_folder)
    filter_by_size(cleaned_folder)
    remove_duplicates(cleaned_folder)
    flag_watermarks(cleaned_folder)

    after = len(iter_images(cleaned_folder))

    logger.info(f"Images after cleaning: {after} (Deleted: {before - after})")

    manual_review_grid(cleaned_folder)


def main():
    logger.info("Cleaning started")

    for species in SPECIES:
        clean_species(species)

    logger.info("Cleaning done")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
