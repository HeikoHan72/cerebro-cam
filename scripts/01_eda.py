#!/usr/bin/env python3
"""
SKRIPT 01: EXPLORATIVE DATENANALYSE (EDA)

Untersucht die Rohdaten des Trainings- und Testdatensatzes (Nickparvar):
Klassenverteilung, Bildgroessen, Farbmodi und Beispielbilder.
Voraussetzung: scripts/00_setup_data.py wurde ausgefuehrt.
"""

# %% ============================================================================
# BLOCK 1: IMPORTE & PFADE
# ==============================================================================

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from cerebrocam import config
from cerebrocam.data import list_images

TRAIN_DIR = config.TRAIN_RAW_DIR
TEST_DIR = config.TEST_RAW_DIR
FIGURES_DIR = config.FIGURES_DIR
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = config.CLASSES


def print_header(title: str) -> None:
    print("------------------------------------------------------------------")
    print(title)
    print("------------------------------------------------------------------")


# %% ============================================================================
# BLOCK 2: PFADPRUEFUNG
# Beschreibung: Prueft, ob die Ordner 'Training' und 'Testing' vorhanden sind.
# ==============================================================================


def check_paths():
    print_header("BLOCK 2: Pfadpruefung")
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Datenordner nicht gefunden: {TRAIN_DIR} / {TEST_DIR}\n"
            "Bitte zuerst scripts/00_setup_data.py ausfuehren."
        )
    print(f"Ordner gefunden: {TRAIN_DIR}")
    print(f"Ordner gefunden: {TEST_DIR}\n")


# %% ============================================================================
# BLOCK 3: KLASSENVERTEILUNG ZAEHLEN
# Beschreibung: Zaehlt alle Bilddateien pro Klasse in 'Training' und 'Testing'.
# ==============================================================================


def get_class_counts(data_dir: Path) -> dict:
    return {cls: len(list_images(data_dir / cls)) for cls in CLASSES}


def analyze_distributions():
    print_header("BLOCK 3: Klassenverteilung")
    train_counts = get_class_counts(TRAIN_DIR)
    test_counts = get_class_counts(TEST_DIR)

    for name, counts in [("Training", train_counts), ("Test", test_counts)]:
        print(f"{name}-Datensatz:")
        for cls, count in counts.items():
            print(f"  {cls:12s}: {count:5d} Bilder")
        print(f"  Gesamt        : {sum(counts.values())} Bilder\n")
    return train_counts, test_counts


# %% ============================================================================
# BLOCK 4: BILDEIGENSCHAFTEN
# Beschreibung: Untersucht Aufloesungen und Farbmodi aller Trainingsbilder.
# ==============================================================================


def inspect_image_properties(data_dir: Path):
    print_header("BLOCK 4: Bildeigenschaften (Groessen und Farbmodi)")
    sizes, modes = [], []
    for cls in CLASSES:
        for img_path in list_images(data_dir / cls):
            with Image.open(img_path) as img:
                sizes.append(img.size)
                modes.append(img.mode)

    size_counts = Counter(sizes)
    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]
    print(f"Untersuchte Bilder: {len(sizes)}, verschiedene Groessen: {len(size_counts)}")
    print(f"Breite: {min(widths)}-{max(widths)} Pixel, Hoehe: {min(heights)}-{max(heights)} Pixel")
    print("Haeufigste Groessen (Breite x Hoehe):")
    for (w, h), count in size_counts.most_common(5):
        print(f"  {w}x{h} Pixel: {count} Bilder")
    print("\nFarbmodi:")
    for mode, count in Counter(modes).most_common():
        print(f"  Modus '{mode}': {count} Bilder")
    print()


# %% ============================================================================
# BLOCK 5: VISUALISIERUNG DER KLASSENVERTEILUNG
# Beschreibung: Balkendiagramm Training gegen Test.
# ==============================================================================


def plot_distributions(train_counts: dict, test_counts: dict):
    print_header("BLOCK 5: Verteilungsdiagramm")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, counts, title, palette in [
        (axes[0], train_counts, "Class distribution - training set", "Blues_d"),
        (axes[1], test_counts, "Class distribution - test set", "Greens_d"),
    ]:
        names, values = list(counts.keys()), list(counts.values())
        sns.barplot(x=names, y=values, hue=names, palette=palette, legend=False, ax=ax)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of images")
        ax.set_xlabel("Class")
        for i, v in enumerate(values):
            ax.text(i, v + max(values) * 0.02, str(v), ha="center", fontweight="bold")

    plt.tight_layout()
    out = FIGURES_DIR / "eda_class_distribution.png"
    plt.savefig(out, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Diagramm gespeichert: {out}\n")


# %% ============================================================================
# BLOCK 6: BEISPIELBILDER (GRID)
# Beschreibung: Zeigt die ersten Bilder jeder Klasse (feste, sortierte Auswahl).
# ==============================================================================


def plot_sample_grid(data_dir: Path, samples_per_class: int = 4):
    print_header("BLOCK 6: Beispielbilder")
    fig, axes = plt.subplots(len(CLASSES), samples_per_class, figsize=(12, 10))
    for row, cls in enumerate(CLASSES):
        images = list_images(data_dir / cls)[:samples_per_class]
        for col in range(samples_per_class):
            ax = axes[row, col]
            if col < len(images):
                with Image.open(images[col]) as img:
                    ax.imshow(img.copy(), cmap="gray" if img.mode == "L" else None)
                    ax.set_title(f"{cls}\n({img.size[0]}x{img.size[1]})", fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    out = FIGURES_DIR / "eda_sample_grid.png"
    plt.savefig(out, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Beispiel-Grid gespeichert: {out}\n")


# %% ============================================================================
# BLOCK 7: HAUPTAUSFUEHRUNG
# ==============================================================================


def main():
    print("\n==================================================================")
    print("EDA (PHASE 1: EXPLORATIVE DATENANALYSE)")
    print("==================================================================\n")
    check_paths()
    train_counts, test_counts = analyze_distributions()
    inspect_image_properties(TRAIN_DIR)
    plot_distributions(train_counts, test_counts)
    plot_sample_grid(TRAIN_DIR)
    print("EDA abgeschlossen.")


if __name__ == "__main__":
    main()
