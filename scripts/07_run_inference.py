#!/usr/bin/env python3
"""
SKRIPT 07: INFERENZ AUF DEN ANONYMISIERTEN EXTERNEN BILDERN

Das Modell klassifiziert alle Bilder aus data/external/unlabeled/, ohne die
wahren Klassen zu kennen. Die Bilder werden genauso eingelesen wie beim
Training und bei der internen Evaluierung.
"""

# %% ============================================================================
# BLOCK 1: PFADE & MODELL
# ==============================================================================

import tensorflow as tf

from cerebrocam import config
from cerebrocam.inference import predict_folder

INFERENCE_DIR = config.EXTERNAL_UNLABELED_DIR
if not INFERENCE_DIR.exists():
    raise FileNotFoundError(
        f"Inferenz-Ordner nicht gefunden: {INFERENCE_DIR}. "
        "Bitte zuerst 06_prepare_external_data.py ausfuehren."
    )

model = tf.keras.models.load_model(config.MODEL_PATH)
print(f"Modell geladen: {config.MODEL_PATH}")

# %% ============================================================================
# BLOCK 2: VORHERSAGEN & SPEICHERN
# ==============================================================================

predictions = predict_folder(model, INFERENCE_DIR).rename(columns={"file": "new_filename"})
config.PREDICTIONS_EXTERNAL_CSV.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(config.PREDICTIONS_EXTERNAL_CSV, index=False)

print(f"{len(predictions)} Bilder klassifiziert.")
print(f"Vorhersagen gespeichert: {config.PREDICTIONS_EXTERNAL_CSV}")
