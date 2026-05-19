import json
import logging
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

RAW_DIR = Path("../data/raw")
CLEANED_DIR = Path("../data/cleaned")
FINAL_DIR = Path("../data/final")
DATASET_DIR = Path("../dataset_final")

LOG_DIR = Path("../logs")
REPORT_DIR = Path("../reports")

SPECIES = {
    "hibou_grand-duc": "Hibou grand-duc",
    "flamant_rose": "Flamant rose",
    "martin-pecheur": "Martin-pêcheur",
    "cygne_tubercule": "Cygne tuberculé",
    "pic_vert": "Pic vert",
}

SPLITS = ["train", "val", "test"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
    }
)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0

    return sum(
        1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def parse_cleaning_logs() -> dict[str, dict[str, int]]:
    reasons = {
        slug: {
            "doublons": 0,
            "taille": 0,
            "corruption": 0,
            "filigrane": 0,
        }
        for slug in SPECIES
    }

    for log_path in sorted(LOG_DIR.glob("cleaning_*.log")):
        current_species = None

        try:
            with open(log_path, encoding="utf-8") as f:
                for line in f:

                    for slug in SPECIES:
                        if slug in line.lower():
                            current_species = slug
                            break

                    if current_species is None:
                        continue

                    low = line.lower()

                    if "duplicate" in low or "doublon" in low:
                        reasons[current_species]["doublons"] += 1

                    elif "small" in low or "taille" in low:
                        reasons[current_species]["taille"] += 1

                    elif "corrupted" in low or "corrompu" in low:
                        reasons[current_species]["corruption"] += 1

                    elif "watermark" in low or "filigrane" in low:
                        reasons[current_species]["filigrane"] += 1

        except OSError:
            pass

    return reasons


def collect_stats() -> dict:
    stats = {}

    removal_reasons = parse_cleaning_logs()

    for slug, label in SPECIES.items():

        raw = count_images(RAW_DIR / slug)

        cleaned = count_images(CLEANED_DIR / slug)

        final = count_images(FINAL_DIR / slug)

        removed = max(raw - cleaned, 0)

        reasons = removal_reasons.get(slug, {})

        reasons_total = sum(reasons.values())

        if reasons_total > 0 and reasons_total != removed:
            scale = removed / reasons_total
            reasons = {k: round(v * scale) for k, v in reasons.items()}

        split_counts = {}

        for split in SPLITS:
            split_counts[split] = count_images(DATASET_DIR / split / slug)

        stats[slug] = {
            "label": label,
            "raw": raw,
            "cleaned": cleaned,
            "final": final,
            "removed": removed,
            "removed_pct": round((removed / raw) * 100, 1) if raw else 0,
            "reasons": reasons,
            "splits": split_counts,
            "total_dataset": sum(split_counts.values()),
        }

    return stats


def print_text_report(stats: dict):

    sep = "═" * 74
    thin = "─" * 74

    lines = [
        sep,
        f"{'RAPPORT STATISTIQUE DU DATASET':^74}",
        f"{'Généré le ' + datetime.now().strftime('%d/%m/%Y %H:%M'):^74}",
        sep,
        "",
        "1. ÉVOLUTION DU DATASET",
        thin,
        f"{'Espèce':<25} {'Brutes':>8} {'Nettoyées':>12} {'Finales':>10} {'Suppr.':>10} {'%':>6}",
        thin,
    ]

    for s in stats.values():
        lines.append(
            f"{s['label']:<25} "
            f"{s['raw']:>8} "
            f"{s['cleaned']:>12} "
            f"{s['final']:>10} "
            f"{s['removed']:>10} "
            f"{s['removed_pct']:>5.1f}%"
        )

    lines += [thin, ""]

    lines += [
        "2. MOTIFS DE SUPPRESSION",
        thin,
        f"{'Espèce':<25} {'Doublons':>10} {'Taille':>10} {'Corruption':>12} {'Filigrane':>12}",
        thin,
    ]

    for s in stats.values():
        r = s["reasons"]

        lines.append(
            f"{s['label']:<25} "
            f"{r.get('doublons',0):>10} "
            f"{r.get('taille',0):>10} "
            f"{r.get('corruption',0):>12} "
            f"{r.get('filigrane',0):>12}"
        )

    lines += [thin, ""]

    lines += [
        "3. RÉPARTITION TRAIN / VAL / TEST",
        thin,
        f"{'Espèce':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>10}",
        thin,
    ]

    for s in stats.values():
        sp = s["splits"]

        lines.append(
            f"{s['label']:<25} "
            f"{sp.get('train',0):>8} "
            f"{sp.get('val',0):>8} "
            f"{sp.get('test',0):>8} "
            f"{s['total_dataset']:>10}"
        )

    lines += [thin, ""]

    tot_raw = sum(s["raw"] for s in stats.values())
    tot_cleaned = sum(s["cleaned"] for s in stats.values())
    tot_final = sum(s["final"] for s in stats.values())
    tot_removed = sum(s["removed"] for s in stats.values())

    tot_train = sum(s["splits"]["train"] for s in stats.values())
    tot_val = sum(s["splits"]["val"] for s in stats.values())
    tot_test = sum(s["splits"]["test"] for s in stats.values())

    total_dataset = sum(s["total_dataset"] for s in stats.values())

    lines += [
        "4. TOTAUX GLOBAUX",
        thin,
        f"Images brutes collectées     : {tot_raw}",
        f"Images après nettoyage       : {tot_cleaned}",
        f"Images finales               : {tot_final}",
        f"Images supprimées            : {tot_removed}",
        "",
        f"Dataset final                : {total_dataset}",
        f"  → Train : {tot_train} ({round((tot_train/total_dataset)*100,1) if total_dataset else 0}%)",
        f"  → Val   : {tot_val} ({round((tot_val/total_dataset)*100,1) if total_dataset else 0}%)",
        f"  → Test  : {tot_test} ({round((tot_test/total_dataset)*100,1) if total_dataset else 0}%)",
        sep,
    ]

    report_text = "\n".join(lines)

    print(report_text)

    txt_path = REPORT_DIR / "rapport_statistique.txt"

    txt_path.write_text(report_text, encoding="utf-8")

    logger.info(f"Rapport texte sauvegardé → {txt_path}")

    json_path = REPORT_DIR / "stats.json"

    json_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(f"Stats JSON sauvegardées → {json_path}")


def fig1_dataset_evolution(stats: dict):

    labels = [s["label"] for s in stats.values()]

    raw_vals = [s["raw"] for s in stats.values()]
    cleaned_vals = [s["cleaned"] for s in stats.values()]
    final_vals = [s["final"] for s in stats.values()]

    x = np.arange(len(labels))

    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))

    b1 = ax.bar(
        x - width,
        raw_vals,
        width,
        label="Brutes",
        color=PALETTE[0],
    )

    b2 = ax.bar(
        x,
        cleaned_vals,
        width,
        label="Nettoyées",
        color=PALETTE[2],
    )

    b3 = ax.bar(
        x + width,
        final_vals,
        width,
        label="Finales",
        color=PALETTE[3],
    )

    ax.set_title(
        "Évolution du dataset",
        fontsize=14,
        pad=12,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")

    ax.set_ylabel("Nombre d'images")

    ax.legend()

    ax.bar_label(b1, padding=2, fontsize=8)
    ax.bar_label(b2, padding=2, fontsize=8)
    ax.bar_label(b3, padding=2, fontsize=8)

    plt.tight_layout()

    out = REPORT_DIR / "fig1_dataset_evolution.png"

    plt.savefig(out, bbox_inches="tight")

    plt.show()

    logger.info(f"Figure 1 sauvegardée → {out}")


def fig2_removal_reasons(stats: dict):

    labels = [s["label"] for s in stats.values()]

    reason_keys = [
        "doublons",
        "taille",
        "corruption",
        "filigrane",
    ]

    reason_labels = [
        "Doublons",
        "Taille",
        "Corruption",
        "Filigrane",
    ]

    data = {k: [s["reasons"].get(k, 0) for s in stats.values()] for k in reason_keys}

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 5))

    bottoms = np.zeros(len(labels))

    for i, (key, label) in enumerate(zip(reason_keys, reason_labels)):

        vals = np.array(data[key])

        ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=label,
            color=PALETTE[i],
        )

        bottoms += vals

    ax.set_title(
        "Ventilation des suppressions",
        fontsize=14,
        pad=12,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")

    ax.set_ylabel("Images supprimées")

    ax.legend()

    plt.tight_layout()

    out = REPORT_DIR / "fig2_suppressions.png"

    plt.savefig(out, bbox_inches="tight")

    plt.show()

    logger.info(f"Figure 2 sauvegardée → {out}")


def fig3_split_distribution(stats: dict):

    labels = [s["label"] for s in stats.values()]

    train_vals = [s["splits"]["train"] for s in stats.values()]
    val_vals = [s["splits"]["val"] for s in stats.values()]
    test_vals = [s["splits"]["test"] for s in stats.values()]

    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(
        y,
        train_vals,
        label="Train",
        color=PALETTE[0],
    )

    ax.barh(
        y,
        val_vals,
        left=train_vals,
        label="Val",
        color=PALETTE[1],
    )

    ax.barh(
        y,
        test_vals,
        left=np.add(train_vals, val_vals),
        label="Test",
        color=PALETTE[2],
    )

    ax.set_title(
        "Répartition Train / Val / Test",
        fontsize=14,
        pad=12,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Nombre d'images")

    ax.legend()

    plt.tight_layout()

    out = REPORT_DIR / "fig3_splits.png"

    plt.savefig(out, bbox_inches="tight")

    plt.show()

    logger.info(f"Figure 3 sauvegardée → {out}")


def fig4_pipeline(stats: dict):

    labels = [s["label"] for s in stats.values()]

    stages = {
        "Brutes": [s["raw"] for s in stats.values()],
        "Nettoyées": [s["cleaned"] for s in stats.values()],
        "Finales": [s["final"] for s in stats.values()],
        "Dataset": [s["total_dataset"] for s in stats.values()],
    }

    x = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (label, color) in enumerate(zip(labels, PALETTE)):

        values = [v[i] for v in stages.values()]

        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.5,
            color=color,
            label=label,
        )

    ax.set_title(
        "Évolution du pipeline",
        fontsize=14,
        pad=12,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(list(stages.keys()))

    ax.set_ylabel("Nombre d'images")

    ax.legend(fontsize=9)

    plt.tight_layout()

    out = REPORT_DIR / "fig4_pipeline.png"

    plt.savefig(out, bbox_inches="tight")

    plt.show()

    logger.info(f"Figure 4 sauvegardée → {out}")


def main():

    logger.info("Génération du rapport statistique")

    stats = collect_stats()

    print_text_report(stats)

    logger.info("Génération des visualisations")

    fig1_dataset_evolution(stats)

    fig2_removal_reasons(stats)

    fig3_split_distribution(stats)

    fig4_pipeline(stats)

    logger.info(f"Rapport complet → {REPORT_DIR}")


if __name__ == "__main__":
    main()
