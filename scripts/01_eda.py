#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:02:11 2026

@author: h
"""

#%% ============================================================================
# BLOCK 1: IMPORTE & ABSOLUTER SKRIPTPFAD
# Beschreibung: Zwingt Python, exakt im Speicherort dieses Skripts zu suchen.
# ==============================================================================

import os
from pathlib import Path
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from cerebrocam import config

# Pfade und Klassen aus der zentralen Konfiguration
TRAIN_DIR = config.TRAIN_RAW_DIR
TEST_DIR = config.TEST_RAW_DIR
FIGURES_DIR = config.FIGURES_DIR
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = config.CLASSES

#%% ============================================================================
# BLOCK 2: PFADPRÜFUNG
# Beschreibung: Prüft direkt, ob die Unterordner 'training' und 'test' vorhanden sind.
# ==============================================================================

def check_paths():
    print("------------------------------------------------------------------")
    print("🔍 BLOCK 2: Pfadprüfung")
    print("------------------------------------------------------------------")
    
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            f"❌ Datenordner nicht gefunden: {TRAIN_DIR} / {TEST_DIR}\n"
            "Bitte zuerst scripts/00_setup_data.py ausführen."
        )
        
    print("✅ Ordner 'training' gefunden.")
    print("✅ Ordner 'test' gefunden.\n")


#%% ============================================================================
# BLOCK 3: KLASSENVERTEILUNG ZÄHLEN
# Beschreibung: Zählt alle Bilddateien pro Kategorie in 'training' und 'test'.
# ==============================================================================

def get_class_counts(data_dir: Path):
    counts = {}
    valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}
    
    for cls in CLASSES:
        cls_folder = data_dir / cls
        if cls_folder.exists():
            images = [f for f in cls_folder.iterdir() if f.suffix.lower() in valid_extensions]
            counts[cls] = len(images)
        else:
            counts[cls] = 0
    return counts


def analyze_distributions():
    print("------------------------------------------------------------------")
    print("📊 BLOCK 3: Klassenverteilung (Class Counts)")
    print("------------------------------------------------------------------")
    
    train_counts = get_class_counts(TRAIN_DIR)
    test_counts = get_class_counts(TEST_DIR)
    
    print("Training-Datensatz:")
    for cls, count in train_counts.items():
        print(f"  • {cls:12s}: {count:5d} Bilder")
    print(f"  Gesamt Training : {sum(train_counts.values())} Bilder\n")
    
    print("Test-Datensatz:")
    for cls, count in test_counts.items():
        print(f"  • {cls:12s}: {count:5d} Bilder")
    print(f"  Gesamt Test     : {sum(test_counts.values())} Bilder\n")
    
    return train_counts, test_counts


#%% ============================================================================
# BLOCK 4: BILDEIGENSCHAFTEN UND FORMATAUSWERTUNG
# Beschreibung: Untersucht Auflösungen und Farbkanäle einer Stichprobe.
# ==============================================================================

def inspect_image_properties(data_dir: Path, samples_per_class=50):
    print("------------------------------------------------------------------")
    print("🖼️  BLOCK 4: Bildeigenschaften (Dimensions & Channels)")
    print("------------------------------------------------------------------")
    
    sizes = []
    modes = []
    
    for cls in CLASSES:
        cls_folder = data_dir / cls
        if not cls_folder.exists():
            continue
            
        images = list(cls_folder.glob("*.*"))[:samples_per_class]
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    sizes.append(img.size)
                    modes.append(img.mode)
            except Exception as e:
                print(f"  ⚠️ Fehler beim Lesen von {img_path.name}: {e}")
                
    size_counts = Counter(sizes)
    mode_counts = Counter(modes)
    
    print("Häufigste Bildgrößen (Breite x Höhe):")
    for size, count in size_counts.most_common(5):
        print(f"  • {size[0]}x{size[1]} Pixel: {count} Bilder")
        
    print("\nErkannte Farbkanal-Modi:")
    for mode, count in mode_counts.items():
        print(f"  • Modus '{mode}': {count} Bilder")
    print()


#%% ============================================================================
# BLOCK 5: VISUALISIERUNG DER KLASSENVERTEILUNG
# Beschreibung: Erstellt ein vergleichendes Balkendiagramm (Train vs. Test).
# ==============================================================================

def plot_distributions(train_counts, test_counts):
    print("------------------------------------------------------------------")
    print("📈 BLOCK 5: Erstellung des Verteilungs-Diagramms")
    print("------------------------------------------------------------------")
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.barplot(x=list(train_counts.keys()), y=list(train_counts.values()), ax=ax[0], palette="Blues_d")
    ax[0].set_title("Klassenverteilung im Training-Set", fontsize=12, fontweight='bold')
    ax[0].set_ylabel("Anzahl Bilder")
    ax[0].set_xlabel("Kategorie")
    for i, v in enumerate(train_counts.values()):
        ax[0].text(i, v + (max(train_counts.values()) * 0.02), str(v), ha='center', fontweight='bold')
        
    sns.barplot(x=list(test_counts.keys()), y=list(test_counts.values()), ax=ax[1], palette="Greens_d")
    ax[1].set_title("Klassenverteilung im Test-Set", fontsize=12, fontweight='bold')
    ax[1].set_ylabel("Anzahl Bilder")
    ax[1].set_xlabel("Kategorie")
    for i, v in enumerate(test_counts.values()):
        ax[1].text(i, v + (max(test_counts.values()) * 0.02), str(v), ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_class_distribution.png", dpi=300)
    plt.show()
    print("✅ Diagramm 'eda_class_distribution.png' gespeichert.\n")


#%% ============================================================================
# BLOCK 6: VISUALISIERUNG VON BEISPIELBILDERN (GRID)
# Beschreibung: Erstellt ein Grid mit MRT-Scans aus jeder Klasse.
# ==============================================================================

def plot_sample_grid(data_dir: Path, samples_per_class=4):
    print("------------------------------------------------------------------")
    print("🎨 BLOCK 6: Visualisierung von Beispielbildern (Image Grid)")
    print("------------------------------------------------------------------")
    
    fig, axes = plt.subplots(len(CLASSES), samples_per_class, figsize=(12, 10))
    
    for row_idx, cls in enumerate(CLASSES):
        cls_folder = data_dir / cls
        images = list(cls_folder.glob("*.*"))[:samples_per_class]
        
        for col_idx in range(samples_per_class):
            ax = axes[row_idx, col_idx]
            if col_idx < len(images):
                img_path = images[col_idx]
                img = Image.open(img_path)
                ax.imshow(img, cmap="gray" if img.mode == "L" else None)
                ax.set_title(f"{cls}\n({img.size[0]}x{img.size[1]})", fontsize=9)
            ax.axis("off")
            
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_sample_grid.png", dpi=300)
    plt.show()
    print("✅ Beispiel-Grid 'eda_sample_grid.png' gespeichert.\n")


#%% ============================================================================
# BLOCK 7: HAUPTAUSFÜHRUNG (MAIN ENTRY POINT)
# Beschreibung: Steuert den Ablauf aller EDA-Schritte.
# ==============================================================================

def main():
    print("\n==================================================================")
    print("🚀 STARTE EDA (PHASE 1: EXPLORATIVE DATENANALYSE)")
    print("==================================================================\n")
    
    check_paths()
    train_counts, test_counts = analyze_distributions()
    inspect_image_properties(TRAIN_DIR)
    plot_distributions(train_counts, test_counts)
    plot_sample_grid(TRAIN_DIR)
    
    print("==================================================================")
    print("✅ PHASE 1 (EDA) ERFOLGREICH ABGESCHLOSSEN")
    print("==================================================================\n")


if __name__ == "__main__":
    main()