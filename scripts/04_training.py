#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:07:09 2026

@author: h
"""

#%% ============================================================================
# SCRIPT 03: MODELL-TRAINING (CerebroCAM) - CNN ARCHITEKTUR
# ==============================================================================

import os
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

#%% ============================================================================
# BLOCK 1: KONFIGURATION & PFADE
# ==============================================================================

# Dynamische Pfadermittlung relativ zum Speicherort des Skripts
SCRIPT_DIR = Path(__file__).parent

# Passe diesen Pfad an den Ordner an, in dem deine aufbereiteten Trainingsdaten liegen
DATA_DIR = SCRIPT_DIR / "processed_training"
MODEL_SAVE_PATH = SCRIPT_DIR / "cerebrocam_best_model.keras"
HISTORY_SAVE_PATH = SCRIPT_DIR / "training_history.csv"

# Hyperparameter
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2  # 80% Training, 20% Validierung
SEED = 42

#%% ============================================================================
# BLOCK 2: DATEN LADEN & SPLITTEN
# ==============================================================================

print("==================================================================")
print("📁 LADE DATEN UND ERSTELLE TRAIN/VAL SPLIT")
print("==================================================================\n")

# Trainingsdaten laden (80%)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical' # Für Categorical Crossentropy
)

# WICHTIG: Klassen-Namen sichern, BEVOR das Dataset transformiert wird!
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"\n• Gefundene Klassen: {class_names}\n")

# Validierungsdaten laden (20%)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Performance-Optimierung beim Laden
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

#%% ============================================================================
# BLOCK 3: MODELL-ARCHITEKTUR (SCHLANKES CNN, < 12 SCHICHTEN)
# ==============================================================================

print("\n==================================================================")
print("🧠 BAUE MODELL-ARCHITEKTUR")
print("==================================================================\n")

model = models.Sequential([
    # Input & Data Augmentation (optional, aber empfohlen zur Robustheit)
    layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Rescaling(1./255),
    
    # 1. Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    
    # 2. Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    
    # 3. Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    
    # Klassifikator-Kopf (Flatten & Dense)
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Regularisierung gegen Overfitting
    layers.Dense(num_classes, activation='softmax') # Output Layer
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

#%% ============================================================================
# BLOCK 4: CALLBACKS & TRAINING
# ==============================================================================

print("\n==================================================================")
print("🚀 STARTE TRAINING")
print("==================================================================\n")

# Callbacks definieren
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,          # Stoppt nach 5 Epochen ohne Verbesserung
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = callbacks.ModelCheckpoint(
    filepath=str(MODEL_SAVE_PATH),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Training ausführen
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, model_checkpoint]
)

#%% ============================================================================
# BLOCK 5: HISTORIE SPEICHERN FÜR EVALUIERUNG
# ==============================================================================

print("\n==================================================================")
print("💾 SPEICHERE TRAININGSHISTORIE")
print("==================================================================\n")

# History in einen DataFrame umwandeln und als CSV speichern
hist_df = pd.DataFrame(history.history)
hist_df.to_csv(HISTORY_SAVE_PATH, index=False)

print(f"✅ Training abgeschlossen!")
print(f"✅ Bestes Modell gespeichert unter: {MODEL_SAVE_PATH.name}")
print(f"✅ Trainingsverlauf gespeichert unter: {HISTORY_SAVE_PATH.name}")
print("==================================================================")