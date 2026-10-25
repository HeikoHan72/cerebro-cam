"""Zentrale Konfiguration: Pfade, Klassen, Bildgroesse, Hyperparameter und Datensaetze.

Alle Skripte in scripts/ lesen ihre Einstellungen von hier. Pfade sind relativ
zum Projektordner, damit die Pipeline auf jedem Rechner laeuft.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Projektordner
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Daten (nicht im Repository, siehe scripts/00_setup_data.py)
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # hier die beiden Kaggle-ZIPs ablegen

NICKPARVAR_DIR = DATA_DIR / "nickparvar"  # Trainings- und interner Testdatensatz
TRAIN_RAW_DIR = NICKPARVAR_DIR / "Training"
TEST_RAW_DIR = NICKPARVAR_DIR / "Testing"

SARTAJ_DIR = DATA_DIR / "sartaj"  # externer Datensatz

PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PROCESSED_DIR = PROCESSED_DIR / "training"
TEST_PROCESSED_DIR = PROCESSED_DIR / "testing"

EXTERNAL_DIR = DATA_DIR / "external"
EXTERNAL_UNLABELED_DIR = EXTERNAL_DIR / "unlabeled"  # anonymisierte externe Bilder
EXTERNAL_MAPPING_CSV = EXTERNAL_DIR / "evaluation_mapping.csv"

# ---------------------------------------------------------------------------
# Modell und Ergebnisse
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "cerebrocam_best_model.keras"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
HISTORY_CSV = RESULTS_DIR / "training_history.csv"
TRAINING_SUMMARY_JSON = RESULTS_DIR / "training_summary.json"
PREDICTIONS_INTERNAL_CSV = RESULTS_DIR / "predictions_internal.csv"
PREDICTIONS_EXTERNAL_CSV = RESULTS_DIR / "predictions_external.csv"
METRICS_INTERNAL_JSON = RESULTS_DIR / "metrics_internal.json"
METRICS_EXTERNAL_JSON = RESULTS_DIR / "metrics_external.json"

# ---------------------------------------------------------------------------
# Klassen (alphabetisch, so wie Keras sie aus den Ordnernamen liest)
# ---------------------------------------------------------------------------
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Ordnernamen im externen Datensatz -> Klassennamen des Modells
SARTAJ_LABEL_MAP = {
    "glioma_tumor": "glioma",
    "meningioma_tumor": "meningioma",
    "no_tumor": "notumor",
    "pituitary_tumor": "pituitary",
}

# ---------------------------------------------------------------------------
# Bilder und Training
# ---------------------------------------------------------------------------
IMG_SIZE = (150, 150)  # (Hoehe, Breite) wie bei Keras; PIL erwartet (Breite, Hoehe)
JPEG_QUALITY = 95
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2
SEED = 42
EARLY_STOPPING_PATIENCE = 5

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# Datensaetze: Quelle, Erkennungsmerkmal in der ZIP, Pruefsumme, Bildanzahl
# Die Pruefsummen gehoeren zu den ZIPs, mit denen das Projekt entwickelt wurde.
# ---------------------------------------------------------------------------
DATASETS = {
    "nickparvar": {
        "kaggle": "masoudnickparvar/brain-tumor-mri-dataset",
        "url": "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset",
        "marker": "Training/glioma/",
        "sha256": "882817250048c78ef7a759cf23e540d7b581f2327b16663c9d3db12f5d2ffdb4",
        "target_dir": NICKPARVAR_DIR,
        "expected_counts": {
            "Training": {"glioma": 1400, "meningioma": 1400, "notumor": 1400, "pituitary": 1400},
            "Testing": {"glioma": 400, "meningioma": 400, "notumor": 400, "pituitary": 400},
        },
    },
    "sartaj": {
        "kaggle": "sartajbhuvaji/brain-tumor-classification-mri",
        "url": "https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri",
        "marker": "Training/glioma_tumor/",
        "sha256": "b97a99ee09daf073bfa3e49a5a09002a582b4ee8478a26d3af3a2c67be62c79c",
        "target_dir": SARTAJ_DIR,
        "expected_counts": {
            "Training": {
                "glioma_tumor": 826, "meningioma_tumor": 822, "no_tumor": 395, "pituitary_tumor": 827
            },
            "Testing": {
                "glioma_tumor": 100, "meningioma_tumor": 115, "no_tumor": 105, "pituitary_tumor": 74
            },
        },
    },
}
