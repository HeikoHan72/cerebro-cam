#!/usr/bin/env python3
"""
SKRIPT 03: PREPROCESSING

Wandelt alle Trainings- und Testbilder in RGB, skaliert sie mit dem
Lanczos-Filter auf 150x150 Pixel und speichert sie als JPEG (Qualitaet 95).
Die externe Pipeline (Skript 06) nutzt dieselbe Funktion, damit alle Bilder
identisch vorverarbeitet werden.
"""

# %% ============================================================================
# BLOCK 1: PFADE & KONFIGURATION
# ==============================================================================

from pathlib import Path

from cerebrocam import config
from cerebrocam.data import list_images
from cerebrocam.preprocessing import preprocess_image

SRC_TRAIN = config.TRAIN_RAW_DIR
SRC_TEST = config.TEST_RAW_DIR
DEST_TRAIN = config.TRAIN_PROCESSED_DIR
DEST_TEST = config.TEST_PROCESSED_DIR
CLASSES = config.CLASSES


# %% ============================================================================
# BLOCK 2: VORVERARBEITEN & SPEICHERN
# ==============================================================================

def process_and_save_images(src_root: Path, dest_root: Path):
    """Durchlaeuft alle Klassenordner, skaliert die Bilder und speichert sie ab."""
    print(f"Starte Export nach: {dest_root}")
    for cls in CLASSES:
        src_cls_dir = src_root / cls
        if not src_cls_dir.exists():
            raise FileNotFoundError(
                f"Quellordner nicht gefunden: {src_cls_dir}. Bitte zuerst 00_setup_data.py ausfuehren."
            )
        images = list_images(src_cls_dir)
        print(f"  Klasse '{cls}': {len(images)} Bilder")
        for img_path in images:
            preprocess_image(img_path, dest_root / cls / f"{img_path.stem}.jpg")
    print(f"  Export fuer '{dest_root.name}' abgeschlossen.\n")


# %% ============================================================================
# BLOCK 3: HAUPTAUSFUEHRUNG
# ==============================================================================

def main():
    print("\n==================================================================")
    print("PREPROCESSING")
    print("==================================================================\n")
    process_and_save_images(SRC_TRAIN, DEST_TRAIN)
    process_and_save_images(SRC_TEST, DEST_TEST)
    print("Preprocessing abgeschlossen.")


if __name__ == "__main__":
    main()
