#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:48:06 2026

@author: h
"""

#%% ============================================================================
# SCRIPT 07: EXTERNE EVALUIERUNG (Der finale Proof)
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

#%% ============================================================================
# BLOCK 1: PFADE & CSV-DATEIEN LADEN
# ==============================================================================

print("==================================================================")
print("📊 LADE DATEN FÜR DEN FINALEN ABGLEICH")
print("==================================================================\n")

SCRIPT_DIR = Path(__file__).parent

# Pfade zu den beiden CSV-Dateien (ggf. anpassen, falls sie woanders liegen)
PREDICTIONS_CSV = SCRIPT_DIR / "inference_predictions.csv"
MAPPING_CSV = SCRIPT_DIR / "Datensatz" / "Inferenz und Evaluierungs Daten" / "evaluation_mapping.csv"

if not PREDICTIONS_CSV.exists() or not MAPPING_CSV.exists():
    print("❌ Fehler: Eine der CSV-Dateien wurde nicht gefunden!")
    print(f"Suche Vorhersagen unter: {PREDICTIONS_CSV}")
    print(f"Suche Mapping unter: {MAPPING_CSV}")
    exit()

# Einlesen der CSV-Dateien mit pandas
preds_df = pd.read_csv(PREDICTIONS_CSV)
mapping_df = pd.read_csv(MAPPING_CSV)

# Zusammenführen (Merge) anhand des anonymen Dateinamens ('new_filename')
merged_df = pd.merge(preds_df, mapping_df, on='new_filename')

print(f"• Erfolgreich {len(merged_df)} Bilder abgeglichen.")

#%% ============================================================================
# BLOCK 2: METRIKEN BERECHNEN & AUSGEBEN
# ==============================================================================

print("\n==================================================================")
print("📈 ERGEBNISSE DER EXTERNEN VALIDIERUNG")
print("==================================================================\n")

# Wir standardisieren die echten Labels aus den Ordnernamen des fremden Datensatzes, 
# damit sie exakt zu den Vorhersagen (Klassen) unseres Modells passen.
label_mapping = {
    'glioma_tumor': 'glioma',
    'meningioma_tumor': 'meningioma',
    'no_tumor': 'notumor',
    'pituitary_tumor': 'pituitary',
    # Fallbacks, falls die Namen schon teilweise stimmen:
    'glioma': 'glioma',
    'meningioma': 'meningioma',
    'notumor': 'notumor',
    'pituitary': 'pituitary'
}

# Mapping anwenden
merged_df['true_class'] = merged_df['true_class'].map(lambda x: label_mapping.get(x.lower(), x))

# Wahre und vorhergesagte Labels extrahieren
y_true = merged_df['true_class']
y_pred = merged_df['predicted_class']

# Die 4 Klassennamen fest definieren (müssen exakt so heißen wie im Modell)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Klassifikationsbericht ausgeben
print("KLASSIFIKATIONSBERICHT (EXTERNER DATENSATZ):\n")
report = classification_report(y_true, y_pred, target_names=class_names, labels=class_names)
print(report)


#%% ============================================================================
# BLOCK 3: CONFUSION MATRIX PLOTTEN
# ==============================================================================

# Confusion Matrix berechnen
cm = confusion_matrix(y_true, y_pred, labels=class_names)

# Plotten
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', # Oranges zur Unterscheidung von der internen Evaluierung (Blues)
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Externer Validierungsdatensatz')
plt.xlabel('Vorhergesagte Klasse (KI)')
plt.ylabel('Tatsächliche Klasse (Realität)')
plt.tight_layout()

# Abspeichern für die Präsentation
save_path = SCRIPT_DIR / "confusion_matrix_extern.png"
plt.savefig(save_path, dpi=300)
print(f"✅ Externe Confusion Matrix gespeichert unter: {save_path.name}")

plt.show()
print("==================================================================")