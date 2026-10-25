#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:40:12 2026

@author: h
"""

#%% ============================================================================
# SCRIPT 06: INFERENZ AUSFÜHREN (Auf unklassifizierten Daten)
# ==============================================================================

import csv
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from cerebrocam import config

#%% ============================================================================
# BLOCK 1: PFADE & KONFIGURATION
# ==============================================================================

print("==================================================================")
print("⚙️ INITIALISIERE INFERENZ-UMGEBUNG")
print("==================================================================\n")

# Pfade aus der zentralen Konfiguration
INFERENCE_DIR = config.EXTERNAL_UNLABELED_DIR
MODEL_PATH = config.MODEL_PATH
PREDICTIONS_CSV = config.PREDICTIONS_CSV
PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)

# Die genauen Klassennamen in alphabetischer Reihenfolge (wie beim Training)
CLASS_NAMES = config.CLASSES

if not INFERENCE_DIR.exists():
    print(f"❌ Fehler: Der Inferenz-Ordner wurde nicht gefunden:\n{INFERENCE_DIR}")
    print("Bitte zuerst scripts/06_prepare_external_data.py ausführen.")
    exit()

#%% ============================================================================
# BLOCK 2: MODELL LADEN
# ==============================================================================

print("• Lade das trainierte Modell...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Modell '{MODEL_PATH.name}' erfolgreich geladen.\n")

#%% ============================================================================
# BLOCK 3: BILDER EINLESEN UND VORHERSAGEN
# ==============================================================================

print("==================================================================")
print("🚀 STARTE VORHERSAGEN (INFERENZ)")
print("==================================================================\n")

# Alle JPG-Bilder im Ordner finden und sortieren
image_paths = sorted([p for p in INFERENCE_DIR.glob("*.jpg")])
print(f"• {len(image_paths)} anonymisierte Bilder zur Verarbeitung gefunden.")

results = []
counter = 0

for img_path in image_paths:
    try:
        # Bild laden und auf die beim Training verwendete Größe (150x150) anpassen
        img = image.load_img(img_path, target_size=config.IMG_SIZE)
        
        # Bild in ein Numpy-Array umwandeln
        img_array = image.img_to_array(img)
        
        # Eine zusätzliche Dimension hinzufügen (Batch-Größe von 1 simulieren)
        # Das Modell erwartet (Batch, Height, Width, Channels)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Vorhersage generieren (verbose=0 unterdrückt den Konsolen-Spam)
        prediction_probs = model.predict(img_array, verbose=0)
        
        # Höchste Wahrscheinlichkeit ermitteln
        predicted_index = np.argmax(prediction_probs, axis=1)[0]
        confidence = np.max(prediction_probs)
        predicted_class = CLASS_NAMES[predicted_index]
        
        # Ergebnis in der Liste speichern
        results.append({
            "new_filename": img_path.name,
            "predicted_class": predicted_class,
            "confidence": round(float(confidence), 4) # Auf 4 Nachkommastellen runden
        })
        
        counter += 1
        # Kleines Feedback in der Konsole alle 50 Bilder
        if counter % 50 == 0:
            print(f"  ... {counter}/{len(image_paths)} Bilder verarbeitet.")
            
    except Exception as e:
        print(f"⚠️ Fehler bei Bild {img_path.name}: {e}")

#%% ============================================================================
# BLOCK 4: ERGEBNISSE ALS CSV SPEICHERN
# ==============================================================================

print("\n==================================================================")
print("💾 SPEICHERE VORHERSAGEN")
print("==================================================================\n")

with open(PREDICTIONS_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
    fieldnames = ["new_filename", "predicted_class", "confidence"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"✅ Inferenz abgeschlossen!")
print(f"✅ Vorhersagen gespeichert unter: {PREDICTIONS_CSV.name}")
print("==================================================================")