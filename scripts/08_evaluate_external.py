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
# BLOCK 1: DATEN LADEN & ZUSAMMENFUEHREN
# ==============================================================================

import pandas as pd
from sklearn.metrics import classification_report

from cerebrocam import config
from cerebrocam.evaluation import metrics_by_subset, plot_confusion_matrix, save_json

for path in (config.PREDICTIONS_EXTERNAL_CSV, config.EXTERNAL_MAPPING_CSV, config.EXTERNAL_OVERLAP_CSV):
    if not path.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {path}. Bitte zuerst die Skripte 02, 06 und 07 ausfuehren."
        )

preds_df = pd.read_csv(config.PREDICTIONS_EXTERNAL_CSV)
mapping_df = pd.read_csv(config.EXTERNAL_MAPPING_CSV)
overlap_df = pd.read_csv(config.EXTERNAL_OVERLAP_CSV)

# Entblindung ueber den anonymen Dateinamen, dann Ueberlappungsinfo ueber den Originalpfad
merged_df = preds_df.merge(mapping_df, on="new_filename", how="inner", validate="one_to_one")
merged_df = merged_df.merge(
    overlap_df[["original_path", "category", "eval_not_in_training"]],
    on="original_path",
    how="left",
    validate="one_to_one",
)
if len(merged_df) != len(mapping_df) or merged_df["category"].isna().any():
    raise ValueError(
        "Nicht alle Bilder konnten zugeordnet werden. Bitte die Skripte 02, 06 und 07 erneut ausfuehren."
    )
print(f"{len(merged_df)} Bilder abgeglichen.")

# Ordnernamen (z. B. 'glioma_tumor') auf die Klassennamen des Modells abbilden
merged_df["true_class"] = merged_df["true_class"].map(config.SARTAJ_LABEL_MAP)
if merged_df["true_class"].isna().any():
    raise ValueError("Unbekannte Klassennamen in der Mapping-Tabelle.")

# %% ============================================================================
# BLOCK 2: KENNZAHLEN & KONFUSIONSMATRIZEN
# ==============================================================================

class_names = config.CLASSES
subsets = {
    "all": pd.Series(True, index=merged_df.index),
    "not_in_training": merged_df["eval_not_in_training"].astype(bool),
}
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
