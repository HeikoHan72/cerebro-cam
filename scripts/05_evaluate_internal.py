#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:34:51 2026

@author: h
"""

#%% ============================================================================
# SCRIPT 04: INTERNE EVALUIERUNG (Lernkurven & Confusion Matrix)
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

#%% ============================================================================
# BLOCK 1: PFADE & LERNKURVEN PLOTTEN
# ==============================================================================

print("==================================================================")
print("📊 ERSTELLE LERNKURVEN FÜR DIE DOKUMENTATION")
print("==================================================================\n")

SCRIPT_DIR = Path(__file__).parent
TEST_DIR = SCRIPT_DIR / "processed_testing"
MODEL_PATH = SCRIPT_DIR / "cerebrocam_best_model.keras"
HISTORY_PATH = SCRIPT_DIR / "training_history.csv"

# Historie laden
if HISTORY_PATH.exists():
    history_df = pd.read_csv(HISTORY_PATH)
    epochs = range(1, len(history_df) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Plot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_df['accuracy'], label='Trainings-Accuracy', marker='o')
    plt.plot(epochs, history_df['val_accuracy'], label='Validierungs-Accuracy', marker='o')
    plt.title('Modell-Genauigkeit (Accuracy)')
    plt.xlabel('Epochen')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_df['loss'], label='Trainings-Loss', marker='o')
    plt.plot(epochs, history_df['val_loss'], label='Validierungs-Loss', marker='o')
    plt.title('Modell-Verlust (Loss)')
    plt.xlabel('Epochen')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "lernkurven.png", dpi=300)
    print("✅ Lernkurven wurden als 'lernkurven.png' gespeichert.")
    plt.show()
else:
    print("❌ Keine Trainingshistorie gefunden!")

#%% ============================================================================
# BLOCK 2: TESTDATEN LADEN & VORHERSAGEN MACHEN
# ==============================================================================

print("\n==================================================================")
print("🧠 LADE MODELL UND BERECHNE VORHERSAGEN (TEST-DATEN)")
print("==================================================================\n")

# WICHTIG: shuffle=False ist zwingend erforderlich, damit die echten Labels 
# mit den Modellvorhersagen in der gleichen Reihenfolge bleiben!
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(150, 150),
    batch_size=32,
    label_mode='categorical',
    shuffle=False 
)

class_names = test_dataset.class_names
print(f"• Klassen: {class_names}")

# Modell laden
model = tf.keras.models.load_model(MODEL_PATH)
print("• Bestes Modell erfolgreich geladen.")

# Vorhersagen generieren
print("• Führe Vorhersagen auf dem Testdatensatz aus (das kann einen Moment dauern)...")
predictions = model.predict(test_dataset)

# Höchste Wahrscheinlichkeit als Klasse nehmen (Argmax)
y_pred = np.argmax(predictions, axis=1)

# Wahre Labels extrahieren
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_true = np.argmax(y_true, axis=1)

#%% ============================================================================
# BLOCK 3: CONFUSION MATRIX & KLASSIFIKATIONSBERICHT
# ==============================================================================

print("\n==================================================================")
print("📈 EVALUIERUNGSERGEBNISSE")
print("==================================================================\n")

# Klassifikationsbericht ausgeben
print("KLASSIFIKATIONSBERICHT:\n")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# Confusion Matrix berechnen & plotten
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Interner Testdatensatz')
plt.xlabel('Vorhergesagte Klasse (KI)')
plt.ylabel('Tatsächliche Klasse (Realität)')
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "confusion_matrix.png", dpi=300)
print("\n✅ Confusion Matrix wurde als 'confusion_matrix.png' gespeichert.")
plt.show()