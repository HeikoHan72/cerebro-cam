#!/usr/bin/env python3
"""
SKRIPT 08: EXTERNE EVALUIERUNG (Entblindung)

Fuehrt die Vorhersagen aus Skript 07 mit der Mapping-Tabelle aus Skript 06
zusammen und berechnet die Kennzahlen auf dem externen Datensatz.
"""

# %% ============================================================================
# BLOCK 1: PFADE & DATEN LADEN
# ==============================================================================

import pandas as pd
from sklearn.metrics import classification_report

from cerebrocam import config
from cerebrocam.evaluation import compute_metrics, plot_confusion_matrix, save_json

for path in (config.PREDICTIONS_EXTERNAL_CSV, config.EXTERNAL_MAPPING_CSV):
    if not path.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {path}. Bitte zuerst die Skripte 06 und 07 ausfuehren."
        )

preds_df = pd.read_csv(config.PREDICTIONS_EXTERNAL_CSV)
mapping_df = pd.read_csv(config.EXTERNAL_MAPPING_CSV)

# Zusammenfuehren ueber den anonymen Dateinamen; jede Datei muss genau einmal passen
merged_df = pd.merge(preds_df, mapping_df, on="new_filename", how="inner", validate="one_to_one")
if len(merged_df) != len(mapping_df):
    raise ValueError(f"Nur {len(merged_df)} von {len(mapping_df)} Bildern konnten zugeordnet werden.")
print(f"{len(merged_df)} Bilder abgeglichen.")

# %% ============================================================================
# BLOCK 2: LABELS VEREINHEITLICHEN & KENNZAHLEN
# Die Ordnernamen des externen Datensatzes (z. B. 'glioma_tumor') werden auf
# die Klassennamen des Modells abgebildet.
# ==============================================================================

merged_df["true_class"] = merged_df["true_class"].map(config.SARTAJ_LABEL_MAP)
if merged_df["true_class"].isna().any():
    raise ValueError("Unbekannte Klassennamen in der Mapping-Tabelle.")

class_names = config.CLASSES
print("\nKLASSIFIKATIONSBERICHT (EXTERNER DATENSATZ):\n")
y_true, y_pred = merged_df["true_class"], merged_df["predicted_class"]
print(classification_report(y_true, y_pred, labels=class_names, digits=3))

metrics = compute_metrics(y_true, y_pred, class_names)
save_json(metrics, config.METRICS_EXTERNAL_JSON)

# %% ============================================================================
# BLOCK 3: KONFUSIONSMATRIX
# ==============================================================================

out = config.FIGURES_DIR / "confusion_matrix_external.png"
plot_confusion_matrix(metrics["confusion_matrix"], class_names, "Confusion matrix - external dataset",
                      out, cmap="Oranges")

print(f"Kennzahlen gespeichert: {config.METRICS_EXTERNAL_JSON}")
print(f"Konfusionsmatrix:       {out}")
