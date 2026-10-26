#!/usr/bin/env python3
"""
SKRIPT 05: INTERNE EVALUIERUNG (Lernkurven & Konfusionsmatrix)

Bewertet das trainierte Modell auf dem zurueckgehaltenen Testdatensatz
(Nickparvar 'Testing') in zwei Varianten:
  - official_test:     alle 1600 Testbilder der offiziellen Aufteilung
  - deduplicated_test: ohne Bilder, die (nahezu) identisch im Training vorkommen,
                       und ohne Duplikate innerhalb des Testsatzes (Skript 02)
Die bereinigte Variante ist die massgebliche Kennzahl; die offizielle zeigt,
wie stark Duplikate das Ergebnis beschoenigen.
"""

# %% ============================================================================
# BLOCK 1: PFADE & LERNKURVEN
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import classification_report

from cerebrocam import config
from cerebrocam.evaluation import (
    internal_subsets,
    label_internal_predictions,
    metrics_by_subset,
    plot_confusion_matrix,
    save_json,
)
from cerebrocam.inference import predict_folder

TEST_DIR = config.TEST_PROCESSED_DIR
FIGURES_DIR = config.FIGURES_DIR
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("==================================================================")
print("LERNKURVEN")
print("==================================================================\n")

if config.HISTORY_CSV.exists():
    history_df = pd.read_csv(config.HISTORY_CSV)
    epochs = range(1, len(history_df) + 1)
    best_epoch = int(np.argmin(history_df["val_loss"].to_numpy())) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in [(axes[0], "accuracy", "Model accuracy"), (axes[1], "loss", "Model loss")]:
        ax.plot(epochs, history_df[metric], label=f"Training {metric}", marker="o")
        ax.plot(epochs, history_df[f"val_{metric}"], label=f"Validation {metric}", marker="o")
        ax.axvline(best_epoch, color="grey", linestyle="--", label=f"Best epoch ({best_epoch})")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    out = FIGURES_DIR / "learning_curves.png"
    plt.savefig(out, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Lernkurven gespeichert: {out}")
else:
    print(f"Keine Trainingshistorie gefunden ({config.HISTORY_CSV}), Lernkurven werden uebersprungen.")

# %% ============================================================================
# BLOCK 2: TESTDATEN LADEN & VORHERSAGEN
# ==============================================================================

print("\n==================================================================")
print("VORHERSAGEN AUF DEM INTERNEN TESTDATENSATZ")
print("==================================================================\n")

if not config.NICKPARVAR_MANIFEST_CSV.exists():
    raise FileNotFoundError(
        f"Manifest nicht gefunden: {config.NICKPARVAR_MANIFEST_CSV}. Bitte Skript 02 ausfuehren."
    )

model = tf.keras.models.load_model(config.MODEL_PATH)
print(f"Modell geladen: {config.MODEL_PATH}")

# Vorhersagen fuer alle Testbilder, ergaenzt um wahre Klasse und Duplikat-Status (Skript 02)
predictions = label_internal_predictions(predict_folder(model, TEST_DIR), config.NICKPARVAR_MANIFEST_CSV)
predictions.to_csv(config.PREDICTIONS_INTERNAL_CSV, index=False)

# %% ============================================================================
# BLOCK 3: KENNZAHLEN & KONFUSIONSMATRIZEN
# ==============================================================================

class_names = config.CLASSES
subsets = internal_subsets(predictions)
metrics = metrics_by_subset(predictions, subsets, class_names)
save_json(metrics, config.METRICS_INTERNAL_JSON)

for name, mask in subsets.items():
    sub = predictions[mask]
    print(f"KLASSIFIKATIONSBERICHT - {name} ({len(sub)} Bilder):\n")
    print(classification_report(sub["true_class"], sub["predicted_class"], labels=class_names, digits=3))
    out = FIGURES_DIR / f"confusion_matrix_internal_{name.removesuffix('_test')}.png"
    plot_confusion_matrix(
        metrics[name]["confusion_matrix"],
        class_names,
        f"Confusion matrix - internal {name.replace('_', ' ')} (n={len(sub)})",
        out,
        cmap="Blues",
    )

print(f"Accuracy offizieller Testsatz:  {metrics['official_test']['accuracy']:.4f}")
print(f"Accuracy bereinigter Testsatz: {metrics['deduplicated_test']['accuracy']:.4f}")
print(f"Kennzahlen gespeichert:  {config.METRICS_INTERNAL_JSON}")
print(f"Vorhersagen gespeichert: {config.PREDICTIONS_INTERNAL_CSV}")
