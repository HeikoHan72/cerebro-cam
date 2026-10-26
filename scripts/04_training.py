#!/usr/bin/env python3
"""
SKRIPT 04: MODELL-TRAINING (CerebroCAM CNN)

Trainiert das CNN auf den bereinigten, vorverarbeiteten Trainingsbildern
(80 % Training, 20 % Validierung). Mit festem Seed und deterministischen
Operationen liefert derselbe Rechner bei jedem Lauf identische Ergebnisse.
Die Trainingslogik liegt in cerebrocam/training.py und wird auch vom
Augmentation-Experiment (Skript 11) genutzt.
"""

# %% ============================================================================
# BLOCK 1: KONFIGURATION
# ==============================================================================

from cerebrocam import config
from cerebrocam.evaluation import save_json
from cerebrocam.training import train_model

print("==================================================================")
print(f"TRAINING (Seed {config.SEED}, Augmentation: {'ja' if config.USE_AUGMENTATION else 'nein'})")
print("==================================================================\n")

# %% ============================================================================
# BLOCK 2: TRAINING
# ==============================================================================

model, hist_df, summary = train_model(
    seed=config.SEED, augment=config.USE_AUGMENTATION, model_path=config.MODEL_PATH
)

# %% ============================================================================
# BLOCK 3: HISTORIE & ZUSAMMENFASSUNG SPEICHERN
# ==============================================================================

config.HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
hist_df.to_csv(config.HISTORY_CSV)
save_json(summary, config.TRAINING_SUMMARY_JSON)

print("\nTraining abgeschlossen.")
print(f"Bilder pro Klasse: {summary['training_images_per_class']}")
print(f"Klassengewichte:   {summary['class_weights']}")
print(f"Beste Epoche (minimaler val_loss): {summary['best_epoch']} von {summary['epochs_run']}")
print(f"val_loss: {summary['best_val_loss']:.4f}, val_accuracy: {summary['val_accuracy_at_best_epoch']:.4f}")
print(f"Modell gespeichert:           {config.MODEL_PATH}")
print(f"Trainingsverlauf gespeichert: {config.HISTORY_CSV}")
print(f"Zusammenfassung gespeichert:  {config.TRAINING_SUMMARY_JSON}")
