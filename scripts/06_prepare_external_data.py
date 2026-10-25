#!/usr/bin/env python3
"""
SKRIPT 06: EXTERNE DATEN AUFBEREITEN (Anonymisierung & Mapping)

Bereitet den externen Datensatz (Sartaj) fuer eine verblindete Inferenz vor:
Alle Bilder werden mit derselben Funktion wie die Trainingsdaten vorverarbeitet,
in zufaelliger, aber reproduzierbarer Reihenfolge durchnummeriert und flach in
einem Ordner abgelegt. Die wahren Klassen stehen nur in der Mapping-CSV.
"""

# %% ============================================================================
# BLOCK 1: PFADE & KONFIGURATION
# ==============================================================================

import csv
import random

from cerebrocam import config
from cerebrocam.data import list_images
from cerebrocam.preprocessing import preprocess_image

SRC_DIR = config.SARTAJ_DIR  # eingerichtet durch 00_setup_data.py
DEST_DIR = config.EXTERNAL_UNLABELED_DIR
MAPPING_CSV = config.EXTERNAL_MAPPING_CSV

if not SRC_DIR.exists():
    raise FileNotFoundError(
        f"Externer Datensatz nicht gefunden: {SRC_DIR}. Bitte zuerst 00_setup_data.py ausfuehren."
    )

# %% ============================================================================
# BLOCK 2: BILDER SAMMELN & REIHENFOLGE FESTLEGEN
# Sortieren macht die Liste rechnerunabhaengig, das Mischen mit festem Seed
# sorgt dafuer, dass die neuen Dateinamen keine Klassenreihenfolge verraten.
# ==============================================================================

image_paths = list_images(SRC_DIR, recursive=True)
random.Random(config.SEED).shuffle(image_paths)
print(f"{len(image_paths)} Bilddateien gefunden.")

# %% ============================================================================
# BLOCK 3: VORVERARBEITEN, ANONYMISIEREN & MAPPING SCHREIBEN
# ==============================================================================

DEST_DIR.mkdir(parents=True, exist_ok=True)

# Schutz vor Altlasten: fremde Dateien im Zielordner wuerden in Skript 07 mitklassifiziert
planned = {f"{i:05d}.jpg" for i in range(1, len(image_paths) + 1)}
unexpected = [p.name for p in DEST_DIR.iterdir() if p.name not in planned]
if unexpected:
    raise RuntimeError(f"{len(unexpected)} unerwartete Dateien in {DEST_DIR} (z. B. {unexpected[0]}). "
                       "Bitte den Ordner leeren und das Skript erneut starten.")

mapping_rows = []
for counter, img_path in enumerate(image_paths, start=1):
    new_filename = f"{counter:05d}.jpg"
    preprocess_image(img_path, DEST_DIR / new_filename)
    mapping_rows.append({
        "new_filename": new_filename,
        # Ordnername = wahre Klasse, z. B. 'glioma_tumor'
        "true_class": img_path.parent.name,
        # relativer Pfad, z. B. 'Training/glioma_tumor/gg (1).jpg'
        "original_path": img_path.relative_to(SRC_DIR).as_posix(),
    })

with open(MAPPING_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=["new_filename", "true_class", "original_path"])
    writer.writeheader()
    writer.writerows(mapping_rows)

print(f"{len(mapping_rows)} Bilder anonymisiert in: {DEST_DIR}")
print(f"Mapping-Tabelle gespeichert: {MAPPING_CSV}")
