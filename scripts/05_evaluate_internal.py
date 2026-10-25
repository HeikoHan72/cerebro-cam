#!/usr/bin/env python3
"""
SKRIPT 05: INTERNE EVALUIERUNG (Lernkurven & Konfusionsmatrix)

Bewertet das trainierte Modell auf dem zurueckgehaltenen Testdatensatz
(Nickparvar 'Testing') und speichert Kennzahlen, Vorhersagen und Abbildungen.
"""

# %% ============================================================================
# BLOCK 1: PFADE & LERNKURVEN
# ==============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import classification_report

from cerebrocam import config
from cerebrocam.evaluation import compute_metrics, plot_confusion_matrix, save_json

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

# shuffle=False ist zwingend: nur so passen Vorhersagen und echte Labels zusammen
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)
class_names = test_dataset.class_names

model = tf.keras.models.load_model(config.MODEL_PATH)
print(f"Modell geladen: {config.MODEL_PATH}")

probabilities = model.predict(test_dataset, verbose=0)
y_pred_idx = np.argmax(probabilities, axis=1)
y_true_idx = np.argmax(np.concatenate([y for _, y in test_dataset], axis=0), axis=1)
y_true = [class_names[i] for i in y_true_idx]
y_pred = [class_names[i] for i in y_pred_idx]

predictions = pd.DataFrame({
    "file": [str(Path(p).relative_to(TEST_DIR)) for p in test_dataset.file_paths],
    "true_class": y_true,
    "predicted_class": y_pred,
    "confidence": probabilities.max(axis=1).round(4),
})
predictions.to_csv(config.PREDICTIONS_INTERNAL_CSV, index=False)

# %% ============================================================================
# BLOCK 3: KENNZAHLEN & KONFUSIONSMATRIX
# ==============================================================================

print("KLASSIFIKATIONSBERICHT (INTERNER TESTDATENSATZ):\n")
print(classification_report(y_true, y_pred, labels=class_names, digits=3))

metrics = compute_metrics(y_true, y_pred, class_names)
save_json(metrics, config.METRICS_INTERNAL_JSON)

plot_confusion_matrix(metrics["confusion_matrix"], class_names, "Confusion matrix - internal test set",
                      FIGURES_DIR / "confusion_matrix_internal.png", cmap="Blues")

print(f"Kennzahlen gespeichert:   {config.METRICS_INTERNAL_JSON}")
print(f"Vorhersagen gespeichert:  {config.PREDICTIONS_INTERNAL_CSV}")
print(f"Konfusionsmatrix:         {FIGURES_DIR / 'confusion_matrix_internal.png'}")
