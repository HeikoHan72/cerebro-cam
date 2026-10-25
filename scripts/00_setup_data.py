#!/usr/bin/env python3
"""
SKRIPT 00: DATEN EINRICHTEN

Voraussetzung: Im Projektordner einmal `pip install -e .` ausfuehren,
damit die Skripte das Paket `cerebrocam` importieren koennen.

Erwartet die beiden Kaggle-Downloads (jeweils 'archive.zip') in data/raw/:
  - Nickparvar: Trainings- und interner Testdatensatz
  - Sartaj:     externer Datensatz
Das Skript erkennt die ZIPs am Inhalt, prueft die Pruefsumme, entpackt sie
nach data/nickparvar/ und data/sartaj/ und kontrolliert die Bildanzahl.
"""

# %% ============================================================================
# BLOCK 1: IMPORTE
# ==============================================================================

import zipfile

from cerebrocam import config
from cerebrocam.data import count_images, counts_match, identify_zip, sha256_of_file

# %% ============================================================================
# BLOCK 2: ZIPS IN data/raw/ FINDEN UND ZUORDNEN
# Beschreibung: Beide Downloads heissen 'archive.zip'. Die Zuordnung erfolgt
# deshalb ueber die Ordnerstruktur in der ZIP, nicht ueber den Dateinamen.
# ==============================================================================

config.RAW_DIR.mkdir(parents=True, exist_ok=True)

found_zips = {}
for zip_path in sorted(config.RAW_DIR.glob("*.zip")):
    name = identify_zip(zip_path)
    if name is None:
        print(f"Unbekannte ZIP, wird ignoriert: {zip_path.name}")
    elif name in found_zips:
        print(f"Zweite ZIP fuer '{name}' gefunden, wird ignoriert: {zip_path.name}")
    else:
        found_zips[name] = zip_path
        print(f"Gefunden: {zip_path.name} -> Datensatz '{name}'")

# %% ============================================================================
# BLOCK 3: DATENSAETZE PRUEFEN UND ENTPACKEN
# Beschreibung: Bereits vollstaendig entpackte Datensaetze werden uebersprungen.
# Eine abweichende Pruefsumme fuehrt nur zu einer Warnung, eine falsche
# Bildanzahl dagegen zum Abbruch.
# ==============================================================================

for name, info in config.DATASETS.items():
    target = info["target_dir"]
    expected = info["expected_counts"]

    if counts_match(count_images(target), expected):
        print(f"\n[{name}] Bereits eingerichtet: {target}")
        continue

    if name not in found_zips:
        raise FileNotFoundError(
            f"[{name}] ZIP fehlt. Bitte von {info['url']} herunterladen "
            f"und in {config.RAW_DIR} ablegen."
        )

    zip_path = found_zips[name]
    print(f"\n[{name}] Pruefe Pruefsumme von {zip_path.name} ...")
    if sha256_of_file(zip_path) == info["sha256"]:
        print(f"[{name}] Pruefsumme stimmt mit der getesteten Version ueberein.")
    else:
        print(
            f"[{name}] WARNUNG: Pruefsumme weicht ab. Kaggle liefert moeglicherweise "
            "eine andere Version des Datensatzes; Ergebnisse koennen abweichen."
        )

    print(f"[{name}] Entpacke nach {target} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)

    actual = count_images(target)
    if not counts_match(actual, expected):
        raise RuntimeError(f"[{name}] Bildanzahl stimmt nicht.\nErwartet: {expected}\nGefunden: {actual}")
    print(f"[{name}] Bildanzahl korrekt.")

# %% ============================================================================
# BLOCK 4: ZUSAMMENFASSUNG
# ==============================================================================

print("\nUebersicht der eingerichteten Daten:")
for name, info in config.DATASETS.items():
    for split, classes in count_images(info["target_dir"]).items():
        total = sum(classes.values())
        detail = ", ".join(f"{cls}: {n}" for cls, n in classes.items())
        print(f"  {name:10s} {split:8s} {total:5d} Bilder ({detail})")
print("\nDaten sind eingerichtet. Weiter mit scripts/01_eda.py.")
