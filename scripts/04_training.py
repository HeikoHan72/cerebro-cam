#!/usr/bin/env python3
"""
SKRIPT 04: MODELL-TRAINING (CerebroCAM CNN)

Trainiert das CNN auf den vorverarbeiteten Trainingsbildern (80 % Training,
20 % Validierung). Mit festem Seed und deterministischen Operationen liefert
derselbe Rechner bei jedem Lauf identische Ergebnisse.
"""

# %% ============================================================================
# BLOCK 1: KONFIGURATION & REPRODUZIERBARKEIT
# ==============================================================================

import platform

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks

from cerebrocam import config
from cerebrocam.evaluation import save_json
from cerebrocam.model import build_model

# Fester Seed fuer Python, NumPy und TensorFlow sowie deterministische Operationen
tf.keras.utils.set_random_seed(config.SEED)
tf.config.experimental.enable_op_determinism()

DATA_DIR = config.TRAIN_PROCESSED_DIR
MODEL_SAVE_PATH = config.MODEL_PATH
HISTORY_SAVE_PATH = config.HISTORY_CSV
SUMMARY_PATH = config.TRAINING_SUMMARY_JSON
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
HISTORY_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Trainingsdaten nicht gefunden: {DATA_DIR}. Bitte zuerst 03_preprocessing.py ausfuehren."
    )

# %% ============================================================================
# BLOCK 2: DATEN LADEN & SPLITTEN
# ==============================================================================

print("==================================================================")
print("LADE DATEN UND ERSTELLE TRAIN/VAL SPLIT")
print("==================================================================\n")

common = dict(
    validation_split=config.VALIDATION_SPLIT,
    seed=config.SEED,
    image_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    label_mode="categorical",  # fuer Categorical Crossentropy
)
train_dataset = tf.keras.utils.image_dataset_from_directory(DATA_DIR, subset="training", **common)
val_dataset = tf.keras.utils.image_dataset_from_directory(DATA_DIR, subset="validation", **common)

# Klassennamen sichern, bevor das Dataset transformiert wird
class_names = train_dataset.class_names
if class_names != config.CLASSES:
    raise ValueError(f"Unerwartete Klassen: {class_names}")
print(f"\nGefundene Klassen: {class_names}\n")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000, seed=config.SEED).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# %% ============================================================================
# BLOCK 3: MODELL-ARCHITEKTUR (SCHLANKES CNN)
# ==============================================================================

model = build_model(num_classes=len(class_names))
model.summary()

# %% ============================================================================
# BLOCK 4: CALLBACKS & TRAINING
# Beide Callbacks ueberwachen dieselbe Metrik (val_loss). Dadurch enthaelt die
# gespeicherte Datei genau die Epoche, die EarlyStopping als beste wiederherstellt.
# ==============================================================================

print("\n==================================================================")
print("STARTE TRAINING")
print("==================================================================\n")

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=config.EARLY_STOPPING_PATIENCE,  # Stopp nach 5 Epochen ohne Verbesserung
    restore_best_weights=True,
    verbose=1,
)
model_checkpoint = callbacks.ModelCheckpoint(
    filepath=str(MODEL_SAVE_PATH),
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=config.EPOCHS,
    callbacks=[early_stopping, model_checkpoint],
)

# %% ============================================================================
# BLOCK 5: HISTORIE & ZUSAMMENFASSUNG SPEICHERN
# ==============================================================================

hist_df = pd.DataFrame(history.history)
hist_df.index = pd.RangeIndex(1, len(hist_df) + 1, name="epoch")
hist_df.to_csv(HISTORY_SAVE_PATH)

best_epoch = int(np.argmin(hist_df["val_loss"].to_numpy())) + 1
summary = {
    "epochs_run": int(len(hist_df)),
    "best_epoch": best_epoch,
    "best_val_loss": float(hist_df.loc[best_epoch, "val_loss"]),
    "val_accuracy_at_best_epoch": float(hist_df.loc[best_epoch, "val_accuracy"]),
    "train_accuracy_at_best_epoch": float(hist_df.loc[best_epoch, "accuracy"]),
    "seed": config.SEED,
    "python": platform.python_version(),
    "tensorflow": tf.__version__,
    "keras": keras.__version__,
}
save_json(summary, SUMMARY_PATH)

print("\nTraining abgeschlossen.")
print(f"Beste Epoche (minimaler val_loss): {best_epoch} von {len(hist_df)}")
print(f"val_loss: {summary['best_val_loss']:.4f}, val_accuracy: {summary['val_accuracy_at_best_epoch']:.4f}")
print(f"Modell gespeichert:          {MODEL_SAVE_PATH}")
print(f"Trainingsverlauf gespeichert: {HISTORY_SAVE_PATH}")
print(f"Zusammenfassung gespeichert:  {SUMMARY_PATH}")
