#!/usr/bin/env python3
"""
SKRIPT 08: EXTERNE EVALUIERUNG (Entblindung)

Fuehrt die Vorhersagen aus Skript 07 mit der Mapping-Tabelle aus Skript 06
zusammen und berechnet die Kennzahlen in zwei Varianten:
  - all:              alle 3264 Bilder des Sartaj-Datensatzes
  - not_in_training:  nur Bilder, die nicht (nahezu) identisch im Trainingssatz
                      vorkommen, jedes Bild nur einmal (Skript 02)
Wichtig: Der Trainingsdatensatz (Nickparvar) enthaelt laut seiner Beschreibung
Teile des Sartaj-Datensatzes. Die Variante 'all' ist deshalb kein unabhaengiger
Test, sondern zeigt nur, wie stark die Ueberschneidung das Ergebnis beschoenigt.
"""

# %% ============================================================================
# BLOCK 1: DATEN LADEN & ENTBLINDEN
# ==============================================================================

import pandas as pd
from sklearn.metrics import classification_report

from cerebrocam import config
from cerebrocam.evaluation import (
    external_subsets,
    label_external_predictions,
    metrics_by_subset,
    plot_confusion_matrix,
    save_json,
)

for path in (config.PREDICTIONS_EXTERNAL_CSV, config.EXTERNAL_MAPPING_CSV, config.EXTERNAL_OVERLAP_CSV):
    if not path.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {path}. Bitte zuerst die Skripte 02, 06 und 07 ausfuehren."
        )

merged_df = label_external_predictions(
    pd.read_csv(config.PREDICTIONS_EXTERNAL_CSV),
    config.EXTERNAL_MAPPING_CSV,
    config.EXTERNAL_OVERLAP_CSV,
    config.SARTAJ_LABEL_MAP,
)
print(f"{len(merged_df)} Bilder abgeglichen.")

# %% ============================================================================
# BLOCK 2: KENNZAHLEN & KONFUSIONSMATRIZEN
# ==============================================================================

class_names = config.CLASSES
subsets = external_subsets(merged_df)
metrics = metrics_by_subset(merged_df, subsets, class_names)
metrics["overlap_categories"] = merged_df["category"].value_counts().to_dict()
save_json(metrics, config.METRICS_EXTERNAL_JSON)

for name, mask in subsets.items():
    sub = merged_df[mask]
    print(f"\nKLASSIFIKATIONSBERICHT - extern, {name} ({len(sub)} Bilder):\n")
    print(
        classification_report(
            sub["true_class"], sub["predicted_class"], labels=class_names, digits=3, zero_division=0
        )
    )
    plot_confusion_matrix(
        metrics[name]["confusion_matrix"],
        class_names,
        f"Confusion matrix - external, {name.replace('_', ' ')} (n={len(sub)})",
        config.FIGURES_DIR / f"confusion_matrix_external_{name}.png",
        cmap="Oranges",
    )

print(f"Accuracy alle Bilder:         {metrics['all']['accuracy']:.4f}")
print(f"Accuracy nicht im Training:   {metrics['not_in_training']['accuracy']:.4f}")
print(f"Kennzahlen gespeichert: {config.METRICS_EXTERNAL_JSON}")
