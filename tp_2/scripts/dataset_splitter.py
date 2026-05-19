import shutil
import random
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

FINALS_DIR = Path("../data/final")
OUTPUT_DIR = Path("../dataset_final")
LOG_DIR = Path("../logs")

SPECIES = [
    "hibou_grand-duc",
    "flamant_rose",
    "martin-pecheur",
    "cygne_tubercule",
    "pic_vert",
]

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}

RANDOM_SEED = 42


LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"split_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def iter_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def split_list(items: list, ratios: dict[str, float], seed: int = RANDOM_SEED):
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)

    split_names = list(ratios.keys())
    split_sizes = []
    cumulative = 0
    for i, name in enumerate(split_names[:-1]):
        size = round(ratios[name] * n)
        split_sizes.append(size)
        cumulative += size
    split_sizes.append(n - cumulative)

    result = {}
    start = 0
    for name, size in zip(split_names, split_sizes):
        result[name] = shuffled[start : start + size]
        start += size

    return result


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    # collect all
    # for each original find augmented versions
    # split at original lvl so aug vers stay with their original
    # copy every thing into the correct split folder
    # then we're returning a dict of the split name and the count of files copied


def split_species(species: str) -> dict[str, int]:
    finals_folder = FINALS_DIR / species

    if not finals_folder.exists():
        logger.warning(f"Final folder not found: {finals_folder} — skipped.")
        return {}

    images = iter_images(finals_folder)

    if not images:
        logger.warning(f"No images in {finals_folder}")
        return {}

    # split directly
    split_data = split_list(images, SPLITS)

    counts: dict[str, int] = defaultdict(int)

    for split_name, files in split_data.items():
        dst_folder = OUTPUT_DIR / split_name / species
        dst_folder.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            copy_file(img_path, dst_folder / img_path.name)
            counts[split_name] += 1

    return dict(counts)


def print_summary():
    logger.info(f"{'DATASET SUMMARY':^68}")

    header = f"{'Species':<22}" + "".join(f"{s:>12}" for s in SPLITS) + f"{'TOTAL':>10}"
    logger.info(header)
    logger.info("─" * 68)

    totals = defaultdict(int)
    for species in SPECIES:
        row = f"{species:<22}"
        species_total = 0
        for split_name in SPLITS:
            folder = OUTPUT_DIR / split_name / species
            count = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
            row += f"{count:>12}"
            totals[split_name] += count
            species_total += count
        row += f"{species_total:>10}"
        logger.info(row)

    total_row = f"{'TOTAL':<22}" + "".join(f"{totals[s]:>12}" for s in SPLITS)
    total_row += f"{sum(totals.values()):>10}"
    logger.info(total_row)


def print_directory_tree():
    logger.info(f"\ndataset_final/")
    for split_name in SPLITS:
        split_dir = OUTPUT_DIR / split_name
        if not split_dir.exists():
            continue
        logger.info(f"├── {split_name}/")
        species_dirs = sorted(split_dir.iterdir())
        for i, sp_dir in enumerate(species_dirs):
            if not sp_dir.is_dir():
                continue
            count = len(list(sp_dir.glob("*.jpg")))
            connector = "└──" if i == len(species_dirs) - 1 else "├──"
            logger.info(f"│   {connector} {sp_dir.name}/  ({count} images)")


def main():
    logger.info("Dataset Split")
    logger.info(f"Splits: { {k: f'{v*100:.0f}%' for k, v in SPLITS.items()} }")
    logger.info(f"Seed: {RANDOM_SEED}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Log: {log_file}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for species in SPECIES:
        logger.info(f"\n{'─'*60}")
        logger.info(f"Splitting : {species}")
        counts = split_species(species)
        if counts:
            for split_name, n in counts.items():
                logger.info(f"  {split_name:<8}: {n} files")

    print_summary()
    print_directory_tree()

    logger.info(f"\nSplit complete → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
