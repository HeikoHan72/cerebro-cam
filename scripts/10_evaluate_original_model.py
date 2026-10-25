#!/usr/bin/env python3
"""
SKRIPT 10: ORIGINALMODELL MIT DER NEUEN AUSWERTUNG BEWERTEN

Bewertet das Modell aus der Projektwoche (August 2026) mit genau derselben
Auswertung wie das neu trainierte Modell: interner Test offiziell und
bereinigt, extern alle Bilder und nur Bilder, die nicht im Training vorkommen.
So entsteht der Vergleich "vorher / nachher" fuer das README.

Voraussetzungen:
  - Skripte 02, 03 und 06 wurden ausgefuehrt
  - Originalmodell liegt unter models/original_model.keras
    (Release-Datei 'original_model.keras')
"""

# %% ============================================================================
# BLOCK 1: MODELL LADEN
# ==============================================================================

import json

import tensorflow as tf

from cerebrocam import config
from cerebrocam.evaluation import (
    external_subsets,
    internal_subsets,
    label_external_predictions,
    label_internal_predictions,
    metrics_by_subset,
    save_json,
)
from cerebrocam.inference import predict_folder

if not config.ORIGINAL_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Originalmodell nicht gefunden: {config.ORIGINAL_MODEL_PATH}. "
        "Bitte die Release-Datei 'original_model.keras' dorthin kopieren."
    )
model = tf.keras.models.load_model(config.ORIGINAL_MODEL_PATH)
print(f"Originalmodell geladen: {config.ORIGINAL_MODEL_PATH}")

# %% ============================================================================
# BLOCK 2: INTERNER TESTSATZ (OFFIZIELL UND BEREINIGT)
# ==============================================================================

internal = label_internal_predictions(
    predict_folder(model, config.TEST_PROCESSED_DIR), config.NICKPARVAR_MANIFEST_CSV
)
internal_metrics = metrics_by_subset(internal, internal_subsets(internal), config.CLASSES)

# %% ============================================================================
# BLOCK 3: EXTERNER DATENSATZ (ALLE UND NICHT IM TRAINING)
# ==============================================================================

external_preds = predict_folder(model, config.EXTERNAL_UNLABELED_DIR).rename(columns={"file": "new_filename"})
external = label_external_predictions(
    external_preds, config.EXTERNAL_MAPPING_CSV, config.EXTERNAL_OVERLAP_CSV, config.SARTAJ_LABEL_MAP
)
external_metrics = metrics_by_subset(external, external_subsets(external), config.CLASSES)

# %% ============================================================================
# BLOCK 4: SPEICHERN & VERGLEICH AUSGEBEN
# ==============================================================================

result = {
    "model": "original_model.keras (project week, August 2026)",
    "internal": internal_metrics,
    "external": external_metrics,
}
save_json(result, config.METRICS_ORIGINAL_MODEL_JSON)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


new_internal = load_json(config.METRICS_INTERNAL_JSON)
new_external = load_json(config.METRICS_EXTERNAL_JSON)

print("\nAccuracy             Originalmodell   neues Modell")
for label, orig, new in [
    ("intern offiziell", internal_metrics["official_test"], (new_internal or {}).get("official_test")),
    (
        "intern bereinigt",
        internal_metrics["deduplicated_test"],
        (new_internal or {}).get("deduplicated_test"),
    ),
    ("extern alle", external_metrics["all"], (new_external or {}).get("all")),
    ("extern nicht im Tr.", external_metrics["not_in_training"], (new_external or {}).get("not_in_training")),
]:
    new_acc = f"{new['accuracy']:.4f}" if new else "  -  "
    print(f"{label:20s} {orig['accuracy']:.4f} (n={orig['n']:4d})   {new_acc}")
print(f"\nGespeichert: {config.METRICS_ORIGINAL_MODEL_JSON}")
