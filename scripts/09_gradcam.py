#!/usr/bin/env python3
"""
SKRIPT 09: GRAD-CAM (Erklaerbarkeit)

Zeigt fuer Bilder des bereinigten Testsatzes, welche Bildbereiche die
Vorhersage des Modells beeinflusst haben. Damit laesst sich pruefen, ob das
Modell auf den Tumor schaut oder auf Nebensaechliches wie Bildraender.

Auswahl der Beispiele (fest, ohne Rosinenpicken): jeweils die ersten Bilder
nach Dateiname.
  - richtig erkannt: 3 Bilder pro Klasse
  - Tumorbilder, die das Modell als 'notumor' einstuft (haeufigster Fehler)
Voraussetzung: Skript 05 wurde ausgefuehrt (results/predictions_internal.csv).
"""

# %% ============================================================================
# BLOCK 1: MODELL & VORHERSAGEN LADEN
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap

from cerebrocam import config
from cerebrocam.gradcam import build_gradcam_model, gradcam, load_image

N_CORRECT_PER_CLASS = 3
N_ERRORS = 8

for path in (config.MODEL_PATH, config.PREDICTIONS_INTERNAL_CSV):
    if not path.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {path}. Bitte zuerst die Skripte 04 und 05 ausfuehren."
        )

model = tf.keras.models.load_model(config.MODEL_PATH)
grad_model = build_gradcam_model(model, layer_name="conv3")
predictions = pd.read_csv(config.PREDICTIONS_INTERNAL_CSV)
predictions = predictions[predictions["keep"]].sort_values("file")  # nur bereinigter Testsatz

# %% ============================================================================
# BLOCK 2: BEISPIELE AUSWAEHLEN
# ==============================================================================

correct = pd.concat(
    [
        predictions[(predictions["true_class"] == c) & (predictions["predicted_class"] == c)].head(
            N_CORRECT_PER_CLASS
        )
        for c in config.CLASSES
    ]
)
missed_tumors = predictions[
    (predictions["true_class"] != "notumor") & (predictions["predicted_class"] == "notumor")
].head(N_ERRORS)
print(f"Beispiele: {len(correct)} richtig erkannt, {len(missed_tumors)} uebersehene Tumoren")

# %% ============================================================================
# BLOCK 3: GRAD-CAM BERECHNEN & ZEICHNEN
# Ueberlagerung in einem einzigen Farbton (Orange): je kraeftiger, desto
# relevanter der Bereich fuer die vorhergesagte Klasse.
# ==============================================================================

HIGHLIGHT = np.array([0xEB, 0x68, 0x34]) / 255  # Orange
MAX_OPACITY = 0.8
MIN_RELEVANCE = 0.2  # schwache Werte (Rauschen) werden nicht eingefaerbt


def overlay(cam: np.ndarray) -> np.ndarray:
    """RGBA-Bild: Orange, dessen Deckkraft mit der Relevanz waechst."""
    alpha = np.clip((cam - MIN_RELEVANCE) / (1 - MIN_RELEVANCE), 0, 1) * MAX_OPACITY
    rgba = np.zeros((*cam.shape, 4))
    rgba[..., :3] = HIGHLIGHT
    rgba[..., 3] = alpha
    return rgba


def plot_examples(examples: pd.DataFrame, ncols: int, title: str, filename: str) -> None:
    nrows = int(np.ceil(len(examples) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 3.3 * nrows + 0.8), squeeze=False, layout="constrained"
    )
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), examples.iterrows(), strict=False):
        image = load_image(config.TEST_PROCESSED_DIR / row["file"])
        cam, class_index, probs = gradcam(grad_model, image)
        predicted = config.CLASSES[class_index]
        if predicted != row["predicted_class"]:
            raise ValueError(
                f"Abweichende Vorhersage fuer {row['file']}: {predicted} statt {row['predicted_class']}"
            )
        ax.imshow(image.mean(axis=-1), cmap="gray", vmin=0, vmax=255)
        ax.imshow(overlay(cam))
        ax.set_title(
            f"true: {row['true_class']}\npredicted: {predicted} ({probs[class_index]:.0%})",
            fontsize=8,
            color="#0b0b0b",
        )
    fig.suptitle(title, fontsize=12)
    legend_cmap = LinearSegmentedColormap.from_list("relevance", ["#1a1a19", "#eb6834"])
    mappable = plt.cm.ScalarMappable(cmap=legend_cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), location="bottom", shrink=0.5, aspect=40)
    cbar.set_label(f"Grad-CAM relevance (values below {MIN_RELEVANCE} not shown)", fontsize=9)
    out = config.FIGURES_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.show()
    plt.close(fig)
    print(f"Gespeichert: {out}")


plot_examples(
    correct, N_CORRECT_PER_CLASS, "Grad-CAM: correctly classified test images", "gradcam_correct.png"
)
if len(missed_tumors):
    plot_examples(
        missed_tumors, 4, "Grad-CAM: tumor images classified as 'notumor'", "gradcam_missed_tumors.png"
    )
