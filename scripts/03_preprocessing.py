#!/usr/bin/env python3
"""
SKRIPT 03: PREPROCESSING

Wandelt die Bilder in RGB, skaliert sie mit dem Lanczos-Filter auf 150x150
Pixel und speichert sie als JPEG (Qualitaet 95). Die externe Pipeline
(Skript 06) nutzt dieselbe Funktion.

Welche Bilder verwendet werden, legt der Duplikat-Check (Skript 02) fest:
  - Training: nur die bereinigten Bilder (keine Duplikate).
  - Test: alle Bilder. Skript 05 wertet den offiziellen und den bereinigten
    Testsatz getrennt aus.
"""

# %% ============================================================================
# BLOCK 1: PFADE & AUSWAHL DER BILDER
# ==============================================================================

from pathlib import Path

import pandas as pd

from cerebrocam import config
from cerebrocam.preprocessing import preprocess_image

if not config.NICKPARVAR_MANIFEST_CSV.exists():
    raise FileNotFoundError(
        f"Manifest nicht gefunden: {config.NICKPARVAR_MANIFEST_CSV}. "
        "Bitte zuerst 02_check_duplicates.py ausfuehren."
    )

manifest = pd.read_csv(config.NICKPARVAR_MANIFEST_CSV)
train_files = manifest.loc[(manifest["split"] == "Training") & manifest["keep"], "file"].tolist()
test_files = manifest.loc[manifest["split"] == "Testing", "file"].tolist()


# %% ============================================================================
# BLOCK 2: VORVERARBEITEN & SPEICHERN
# ==============================================================================


def process_and_save_images(files: list[str], dest_root: Path):
    """Verarbeitet die Bilder (Pfade relativ zu data/nickparvar/) nach dest_root/<Klasse>/."""
    planned = {dest_root / Path(f).parts[1] / f"{Path(f).stem}.jpg": config.NICKPARVAR_DIR / f for f in files}

    # Schutz vor Altlasten aus frueheren Laeufen (z. B. vor dem Duplikat-Check)
    if dest_root.exists():
        unexpected = [p for p in dest_root.rglob("*") if p.is_file() and p not in planned]
        if unexpected:
            raise RuntimeError(
                f"{len(unexpected)} unerwartete Dateien in {dest_root} (z. B. {unexpected[0].name}). "
                "Bitte den Ordner loeschen und das Skript erneut starten."
            )

    print(f"Starte Export nach: {dest_root} ({len(planned)} Bilder)")
    for dest, src in planned.items():
        preprocess_image(src, dest)
    counts = pd.Series([d.parent.name for d in planned]).value_counts().sort_index()
    print("  " + ", ".join(f"{cls}: {n}" for cls, n in counts.items()) + "\n")


# %% ============================================================================
# BLOCK 3: HAUPTAUSFUEHRUNG
# ==============================================================================


def main():
    print("\n==================================================================")
    print("PREPROCESSING")
    print("==================================================================\n")
    process_and_save_images(train_files, config.TRAIN_PROCESSED_DIR)
    process_and_save_images(test_files, config.TEST_PROCESSED_DIR)
    print("Preprocessing abgeschlossen.")


if __name__ == "__main__":
    main()
