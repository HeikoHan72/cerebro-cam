#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:32:13 2026

@author: h
"""

#%% ============================================================================
# BLOCK 1: PFADE & KONFIGURATION
# Beschreibung: Definiert Quell- und Zielordner für den physischen Export.
# ==============================================================================

from pathlib import Path
from PIL import Image

# Ermittelt exakt den Ordner, in dem das Skript liegt
BASE_DIR = Path(__file__).parent.resolve()

SRC_TRAIN = BASE_DIR / "training"
SRC_TEST = BASE_DIR / "testing"

DEST_TRAIN = BASE_DIR / "processed_training"
DEST_TEST = BASE_DIR / "processed_testing"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (150, 150)


#%% ============================================================================
# BLOCK 2: PHYSISCHES PREPROCESSING & EXPORT
# Beschreibung: Skaliert alle Bilder und speichert sie als JPG im Zielordner ab.
# ==============================================================================

def process_and_save_images(src_root: Path, dest_root: Path):
    """Durchläuft alle Klassenordner, skaliert die Bilder und speichert sie ab."""
    print(f"📁 Starte Export nach: {dest_root.name}")
    
    for cls in CLASSES:
        src_cls_dir = src_root / cls
        dest_cls_dir = dest_root / cls
        dest_cls_dir.mkdir(parents=True, exist_ok=True)
        
        if not src_cls_dir.exists():
            print(f"  ⚠️ Quellordner nicht gefunden: {src_cls_dir}")
            continue
            
        images = list(src_cls_dir.glob("*.*"))
        print(f"  • Verarbeite Klasse '{cls}': {len(images)} Bilder...")
        
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    # In RGB konvertieren und auf 150x150 skalieren
                    img_rgb = img.convert("RGB")
                    img_resized = img_rgb.resize(IMG_SIZE, Image.Resampling.LANCZOS)
                    
                    # Physisch auf der Festplatte speichern (als JPEG)
                    dest_path = dest_cls_dir / f"{img_path.stem}.jpg"
                    img_resized.save(dest_path, "JPEG", quality=95)
            except Exception as e:
                print(f"    ❌ Fehler bei {img_path.name}: {e}")
                
    print(f"  ✅ Export für '{dest_root.name}' erfolgreich abgeschlossen.\n")


#%% ============================================================================
# BLOCK 3: HAUPTAUSFÜHRUNG (MAIN ENTRY POINT)
# Beschreibung: Steuert den physischen Export für Train- und Test-Daten.
# ==============================================================================

def main():
    print("\n==================================================================")
    print("🚀 STARTE PHYSISCHES PREPROCESSING (EXPORT AUF FESTPLATTE)")
    print("==================================================================\n")
    
    # 1. Training-Daten verarbeiten und speichern
    process_and_save_images(SRC_TRAIN, DEST_TRAIN)
    
    # 2. Test-Daten verarbeiten und speichern
    process_and_save_images(SRC_TEST, DEST_TEST)
    
    print("==================================================================")
    print("✅ PREPROCESSING ABGESCHLOSSEN: Fertige Daten liegen auf der Platte")
    print("==================================================================\n")


if __name__ == "__main__":
    main()