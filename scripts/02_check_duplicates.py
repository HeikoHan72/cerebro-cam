#!/usr/bin/env python3
"""
SKRIPT 02: DUPLIKAT-CHECK

Findet identische und nahezu identische Bilder per Bild-Fingerabdruck
(pHash + dHash) und legt fest, welche Bilder fuer Training und Test verwendet
werden. Jedes Bild zaehlt nur einmal, und kein Testbild darf im Training
vorkommen:
  - Training: von mehreren Duplikaten bleibt nur das erste (sortiert) erhalten.
  - Test: Bilder mit Duplikat im Training werden ausgeschlossen, Duplikate
    innerhalb des Testsatzes ebenfalls auf ein Bild reduziert.
  - Extern (Sartaj): Jedes Bild wird als 'identical' (im Training enthalten),
    'gray_zone' (meist benachbarte Schicht) oder 'new' eingestuft.

Ergebnis (wird ins Repository aufgenommen, damit die Aufteilung nachpruefbar ist):
  results/duplicates/nickparvar_manifest.csv, external_overlap.csv, summary.json
Laufzeit: etwa 1-2 Minuten.
"""

# %% ============================================================================
# BLOCK 1: IMPORTE & KONFIGURATION
# ==============================================================================

import numpy as np
import pandas as pd

from cerebrocam import config
from cerebrocam.data import list_images
from cerebrocam.duplicates import duplicate_groups, fingerprint_images, nearest_neighbors
from cerebrocam.evaluation import save_json

MAX_DIST = config.DUPLICATE_MAX_DISTANCE
GRAY_MAX = config.GRAY_ZONE_MAX_DISTANCE

for folder in (config.TRAIN_RAW_DIR, config.TEST_RAW_DIR, config.SARTAJ_DIR):
    if not folder.exists():
        raise FileNotFoundError(f"Ordner nicht gefunden: {folder}. Bitte zuerst 00_setup_data.py ausfuehren.")

# %% ============================================================================
# BLOCK 2: FINGERABDRUECKE BERECHNEN
# ==============================================================================

print("Berechne Fingerabdruecke (dauert etwa 1-2 Minuten) ...")
train = fingerprint_images(list_images(config.TRAIN_RAW_DIR, recursive=True), config.NICKPARVAR_DIR)
test = fingerprint_images(list_images(config.TEST_RAW_DIR, recursive=True), config.NICKPARVAR_DIR)
ext = fingerprint_images(list_images(config.SARTAJ_DIR, recursive=True), config.SARTAJ_DIR)
for df in (train, test):
    df["split"] = df["file"].str.split("/").str[0]
    df["class"] = df["file"].str.split("/").str[1]
ext["class"] = ext["file"].str.split("/").str[1].map(config.SARTAJ_LABEL_MAP)
print(f"Training: {len(train)}, Test: {len(test)}, extern: {len(ext)} Bilder")

# %% ============================================================================
# BLOCK 3: TRAINING - DUPLIKATE INNERHALB DES TRAININGSSATZES
# ==============================================================================

train["group"] = duplicate_groups(train, MAX_DIST)
train["keep"] = train["group"] == np.arange(len(train))
train["reason"] = np.where(train["keep"], "kept", "duplicate_within_training")
train["nearest_train_file"] = ""
train["nearest_train_distance"] = np.nan

mixed = train.groupby("group")["class"].nunique().gt(1).sum()
print(
    f"\nTraining: {(~train['keep']).sum()} Duplikate entfernt, {train['keep'].sum()} Bilder bleiben "
    f"(Gruppen mit gemischten Labels: {mixed})"
)

# %% ============================================================================
# BLOCK 4: TEST - DUPLIKATE ZUM TRAINING UND INNERHALB DES TESTSATZES
# Verglichen wird mit allen Trainingsbildern, auch den entfernten Duplikaten.
# ==============================================================================

nn_idx, nn_dist = nearest_neighbors(test, train)
test["nearest_train_file"] = train["file"].to_numpy()[nn_idx]
test["nearest_train_distance"] = nn_dist
test["group"] = duplicate_groups(test, MAX_DIST)

in_training = test["nearest_train_distance"] <= MAX_DIST
first_in_group = test["group"] == np.arange(len(test))
test["keep"] = ~in_training & first_in_group
test["reason"] = np.select(
    [in_training, ~first_in_group], ["duplicate_of_training_image", "duplicate_within_test"], default="kept"
)
print(
    f"Test: {in_training.sum()} mit Duplikat im Training, "
    f"{(~in_training & ~first_in_group).sum()} Duplikate innerhalb des Testsatzes, "
    f"{test['keep'].sum()} bleiben"
)

# %% ============================================================================
# BLOCK 5: EXTERN - UEBERLAPPUNG MIT DEM TRAINING
# Fuer die Auswertung "nicht im Training" zaehlt jedes Bild nur einmal.
# ==============================================================================

nn_idx, nn_dist = nearest_neighbors(ext, train)
ext["nearest_train_file"] = train["file"].to_numpy()[nn_idx]
ext["nearest_train_distance"] = nn_dist
ext["category"] = np.select(
    [nn_dist <= MAX_DIST, nn_dist <= GRAY_MAX], ["identical", "gray_zone"], default="new"
)

ext["eval_not_in_training"] = False
subset = ext.index[ext["category"] != "identical"]
groups = duplicate_groups(ext.loc[subset].reset_index(drop=True), MAX_DIST)
ext.loc[subset, "eval_not_in_training"] = groups == np.arange(len(subset))
print(
    f"Extern: {(ext['category'] == 'identical').sum()} identisch mit Trainingsbildern, "
    f"{(ext['category'] == 'gray_zone').sum()} Grauzone, {(ext['category'] == 'new').sum()} neu; "
    f"{ext['eval_not_in_training'].sum()} eindeutige Bilder fuer die Auswertung 'nicht im Training'"
)

# %% ============================================================================
# BLOCK 6: ERGEBNISSE SPEICHERN
# ==============================================================================

config.DUPLICATES_DIR.mkdir(parents=True, exist_ok=True)
manifest_cols = [
    "split",
    "class",
    "file",
    "group",
    "keep",
    "reason",
    "nearest_train_file",
    "nearest_train_distance",
]
manifest = pd.concat([train, test], ignore_index=True)[manifest_cols]
manifest["nearest_train_distance"] = manifest["nearest_train_distance"].astype("Int64")
manifest.to_csv(config.NICKPARVAR_MANIFEST_CSV, index=False)

ext.rename(columns={"file": "original_path"})[
    [
        "original_path",
        "class",
        "nearest_train_file",
        "nearest_train_distance",
        "category",
        "eval_not_in_training",
    ]
].to_csv(config.EXTERNAL_OVERLAP_CSV, index=False)


def per_class(df: pd.DataFrame, mask: pd.Series | None = None) -> dict:
    """Anzahl Bilder pro Klasse, optional nur fuer die markierten Zeilen."""
    counts = (df if mask is None else df.loc[mask])["class"].value_counts()
    return {c: int(counts.get(c, 0)) for c in config.CLASSES}


summary = {
    "max_distance_identical": MAX_DIST,
    "max_distance_gray_zone": GRAY_MAX,
    "training": {"original": per_class(train), "kept": per_class(train, train["keep"])},
    "test": {
        "original": per_class(test),
        "duplicate_of_training_image": per_class(test, in_training),
        "kept": per_class(test, test["keep"]),
    },
    "external": {
        "total": per_class(ext),
        "identical_to_training": per_class(ext, ext["category"] == "identical"),
        "gray_zone": per_class(ext, ext["category"] == "gray_zone"),
        "new": per_class(ext, ext["category"] == "new"),
        "byte_identical_to_training": int(ext["md5"].isin(set(train["md5"])).sum()),
        "unique_not_in_training": per_class(ext, ext["eval_not_in_training"]),
    },
}
save_json(summary, config.DUPLICATES_SUMMARY_JSON)

print("\nBilder pro Klasse:")
print(
    pd.DataFrame(
        {
            "Train vorher": summary["training"]["original"],
            "Train bereinigt": summary["training"]["kept"],
            "Test vorher": summary["test"]["original"],
            "Test bereinigt": summary["test"]["kept"],
            "Extern gesamt": summary["external"]["total"],
            "Extern nicht im Training": summary["external"]["unique_not_in_training"],
        }
    ).to_string()
)
print(f"\nGespeichert in: {config.DUPLICATES_DIR}")
