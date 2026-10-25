#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:46:36 2026

@author: h
"""

#%% ============================================================================
# SCRIPT: REKURSIVES PREPROCESSING, ANONYMISIERUNG & CSV-MAPPING-ERSTELLUNG
# ==============================================================================

from pathlib import Path
from PIL import Image
import csv

# Exakter Quellpfad
SRC_DIR = Path("/media/h/AI Engineer Study/AI Engineer - Weiterbildung/01_AI Engineer Lehrgang/01_Module Lehrgang/03_Deep Learning/01_Module/5. Woche - Deep Learning Projektwoche/Projektarbeit - CerebroCAM/Datensatz/Inferenz und Evaluierungs Daten")

# Zielordner exakt nach Vorgabe: "Unklassifiziert"
DEST_DIR = SRC_DIR / "Unklassifiziert"

# Pfad für die Mapping-CSV-Datei zur späteren Kontrolle
MAPPING_CSV = SRC_DIR / "evaluation_mapping.csv"


def process_and_create_mapping():
    print("==================================================================")
    print("📁 STARTE PREPROCESSING, ANONYMISIERUNG & MAPPING-ERSTELLUNG")
    print("==================================================================\n")
    
    if not SRC_DIR.exists():
        print(f"❌ Fehler: Der Quellpfad existiert nicht: {SRC_DIR}")
        return

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Unterstützte Bildformate
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    # Rekursive Suche über alle Unterordner hinweg
    image_paths = [
        p for p in SRC_DIR.rglob("*") 
        if p.is_file() and p.suffix.lower() in valid_extensions and DEST_DIR not in p.parents
    ]
    
    print(f"• Insgesamt {len(image_paths)} Bilddateien gefunden.")
    
    mapping_data = []
    counter = 1
    total_processed = 0
    
    for img_path in image_paths:
        try:
            # Echten Ordnernamen als wahre Klasse ermitteln (z.B. glioma, meningioma etc.)
            true_class = img_path.parent.name
            if true_class == "Unklassifiziert":
                continue
                
            # Neuer anonymer Dateiname
            new_filename = f"{counter:05d}.jpg"
            dest_path = DEST_DIR / new_filename
            
            # Bild öffnen, in RGB konvertieren und auf 150x150 skalieren
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize((150, 150), Image.Resampling.LANCZOS)
                img_resized.save(dest_path)
                
            # Mapping-Eintrag für die spätere Kontrolle erfassen
            mapping_data.append({
                "new_filename": new_filename,
                "true_class": true_class,
                "original_filename": img_path.name,
                "original_path": str(img_path)
            })
            
            counter += 1
            total_processed += 1
            
        except Exception as e:
            print(f"⚠️ Warnung bei Datei {img_path.name}: {e}")
            
    # Mapping als CSV-Datei abspeichern
    with open(MAPPING_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["new_filename", "true_class", "original_filename", "original_path"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in mapping_data:
            writer.writerow(row)
            
    print("\n------------------------------------------------------------------")
    print(f"✅ FERTIG! {total_processed} Bilder verarbeitet und nach '{DEST_DIR.name}' kopiert.")
    print(f"📄 Mapping-Tabelle erfolgreich gespeichert unter: {MAPPING_CSV}")
    print("------------------------------------------------------------------\n")


if __name__ == "__main__":
    process_and_create_mapping()