import logging
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, UnidentifiedImageError

SOURCE_DIR = Path("../data/cleaned")
OUTPUT_DIR = Path("../data/cleaned")
LOG_DIR = Path("../logs")

TARGET_SIZE = (224, 224)
JPG_QUALITY = 95  # JPEG compression quality (1–95; 95 = high quality)

NORM_MODE = "zero_one"

SPECIES = [
    "hibou_grand-duc",
    "flamant_rose",
    "martin-pecheur",
    "cygne_tubercule",
    "pic_vert",
]


LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]


def normalize_array(arr: np.ndarray, mode: str = NORM_MODE) -> np.ndarray:
    # normalizes a uint8 numpy array.
    # mode='zero_one'  → values in [0.0, 1.0]
    # mode='minus_one' → values in [-1.0, 1.0]
    arr = arr.astype(np.float32) / 255.0
    if mode == "minus_one":
        arr = arr * 2.0 - 1.0
    return arr


def preprocess_image(
    src_path: Path,
    dst_path: Path,
    target_size: tuple[int, int] = TARGET_SIZE,
    norm_mode: str = NORM_MODE,
    jpg_quality: int = JPG_QUALITY,
) -> bool:

    # steps
    # opens and converts the img to rgb
    # resize to the targeted size
    # normalizes pixel values stored as metadata only saved img stays unit8 for standard JPEG com
    # saves jpg

    try:
        with Image.open(src_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            img_resized = img.resize(target_size, Image.LANCZOS)

            arr = np.array(img_resized, dtype=np.uint8)
            norm_arr = normalize_array(arr, mode=norm_mode)
            if norm_mode == "zero_one":
                assert 0.0 <= norm_arr.min() and norm_arr.max() <= 1.0
            else:
                assert -1.0 <= norm_arr.min() and norm_arr.max() <= 1.0

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            img_resized.save(
                dst_path, format="JPEG", quality=jpg_quality, optimize=True
            )

        return True

    except UnidentifiedImageError:
        logger.warning(f"Format non reconnu: {src_path.name}")
    except OSError as e:
        logger.warning(f"Erreur I/O pour {src_path.name}: {e}")
    except AssertionError:
        logger.warning(f"Valeurs hors plage après normalisation: {src_path.name}")
    except Exception as e:
        logger.warning(f"Erreur inattendue pour {src_path.name}: {e}")

    return False


def preprocess_species(species_slug: str):
    src_folder = SOURCE_DIR / species_slug
    dst_folder = OUTPUT_DIR / species_slug

    if not src_folder.exists():
        logger.warning(f"Dossier source introuvable: {src_folder} alors ignoré.")
        return

    dst_folder.mkdir(parents=True, exist_ok=True)
    images = iter_images(src_folder)

    if not images:
        logger.warning(f"Aucune image dans {src_folder}")
        return

    logger.info(f"Prétraitement : {species_slug}  ({len(images)} images)")
    logger.info(f"Source : {src_folder}")
    logger.info(f"Cible  : {dst_folder}")

    ok = 0
    failed = 0

    for idx, src_path in enumerate(images, start=1):
        # always save with .jpg extension regardless of source format
        dst_path = dst_folder / (src_path.stem + ".jpg")

        success = preprocess_image(src_path, dst_path)
        if success:
            ok += 1
            logger.info(f"  [{idx:>4}/{len(images)}]  {src_path.name}  {dst_path.name}")
        else:
            failed += 1
            logger.error(f"  [{idx:>4}/{len(images)}] Échec: {src_path.name}")

    logger.info(
        f"\nRésumé '{species_slug}': " f"{ok} traitées avec succès, {failed} échecs."
    )


def print_summary():
    logger.info("Résumé final  data/final/")
    total = 0
    for species in SPECIES:
        folder = OUTPUT_DIR / species
        count = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
        total += count
        logger.info(f"  {species:<25} {count:>5} images")
    logger.info(f"{'TOTAL':<25} {total:>5} images")


def main():
    logger.info("Démarrage du prétraitement des images")
    logger.info(f"Taille cible   : {TARGET_SIZE[0]}×{TARGET_SIZE[1]} px")
    logger.info(f"Normalisation  : {NORM_MODE}")
    logger.info(f"Qualité JPEG   : {JPG_QUALITY}")
    logger.info(f"Fichier de log : {log_file}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for species in SPECIES:
        preprocess_species(species)

    print_summary()
    logger.info("\nPrétraitement terminé.")


if __name__ == "__main__":
    main()
