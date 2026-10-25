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

#%% ============================================================================
# BLOCK 1: PFADE & KONFIGURATION
# ==============================================================================

print("==================================================================")
print("⚙️ INITIALISIERE INFERENZ-UMGEBUNG")
print("==================================================================\n")

SCRIPT_DIR = Path(__file__).parent

# Passe diesen Pfad an, falls dein "Unklassifiziert"-Ordner woanders liegt
INFERENCE_DIR = SCRIPT_DIR / "Datensatz" / "Inferenz und Evaluierungs Daten" / "Unklassifiziert"
MODEL_PATH = SCRIPT_DIR / "cerebrocam_best_model.keras"
PREDICTIONS_CSV = SCRIPT_DIR / "inference_predictions.csv"

# Die genauen Klassennamen in alphabetischer Reihenfolge (wie beim Training)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

if not INFERENCE_DIR.exists():
    print(f"❌ Fehler: Der Inferenz-Ordner wurde nicht gefunden:\n{INFERENCE_DIR}")
    print("Bitte den Pfad in Block 1 anpassen.")
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
        img = image.load_img(img_path, target_size=(150, 150))
        
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