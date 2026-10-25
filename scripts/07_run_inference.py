#!/usr/bin/env python3
"""
SKRIPT 07: INFERENZ AUF DEN ANONYMISIERTEN EXTERNEN BILDERN

Das Modell klassifiziert alle Bilder aus data/external/unlabeled/, ohne die
wahren Klassen zu kennen. Die Bilder werden genauso eingelesen wie beim
Training und bei der internen Evaluierung (Keras image_dataset_from_directory).
"""

# %% ============================================================================
# BLOCK 1: PFADE & KONFIGURATION
# ==============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from cerebrocam import config

INFERENCE_DIR = config.EXTERNAL_UNLABELED_DIR
CLASS_NAMES = config.CLASSES  # alphabetisch, wie beim Training

if not INFERENCE_DIR.exists():
    raise FileNotFoundError(
        f"Inferenz-Ordner nicht gefunden: {INFERENCE_DIR}. "
        "Bitte zuerst 06_prepare_external_data.py ausfuehren."
    )

# %% ============================================================================
# BLOCK 2: MODELL & BILDER LADEN
# ==============================================================================

model = tf.keras.models.load_model(config.MODEL_PATH)
print(f"Modell geladen: {config.MODEL_PATH}")

dataset = tf.keras.utils.image_dataset_from_directory(
    INFERENCE_DIR,
    labels=None,  # keine Labels: verblindete Inferenz
    image_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
)

# %% ============================================================================
# BLOCK 3: VORHERSAGEN & SPEICHERN
# ==============================================================================

probabilities = model.predict(dataset, verbose=0)
predictions = pd.DataFrame({
    "new_filename": [Path(p).name for p in dataset.file_paths],
    "predicted_class": [CLASS_NAMES[i] for i in np.argmax(probabilities, axis=1)],
    "confidence": probabilities.max(axis=1).round(4),
})
config.PREDICTIONS_EXTERNAL_CSV.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(config.PREDICTIONS_EXTERNAL_CSV, index=False)

print(f"{len(predictions)} Bilder klassifiziert.")
print(f"Vorhersagen gespeichert: {config.PREDICTIONS_EXTERNAL_CSV}")
