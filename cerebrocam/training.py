"""Training des CNN, gemeinsam genutzt von Skript 04 und dem Augmentation-Experiment (Skript 11)."""

import platform
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks

from cerebrocam import config
from cerebrocam.data import list_images
from cerebrocam.model import build_model


def class_weights_for(data_dir: Path, class_names: list[str]) -> tuple[dict, dict]:
    """Bildanzahl pro Klasse und Klassengewichte N / (K * N_Klasse)."""
    counts = {c: len(list_images(data_dir / c)) for c in class_names}
    total = sum(counts.values())
    weights = {i: total / (len(class_names) * counts[c]) for i, c in enumerate(class_names)}
    return counts, weights


def train_model(
    seed: int = config.SEED,
    augment: bool = config.USE_AUGMENTATION,
    model_path: Path = config.MODEL_PATH,
    data_dir: Path = config.TRAIN_PROCESSED_DIR,
    epochs: int = config.EPOCHS,
    verbose: int = 1,
) -> tuple[keras.Model, pd.DataFrame, dict]:
    """Trainiert das CNN reproduzierbar und speichert die beste Epoche nach model_path.

    seed steuert Gewichtsinitialisierung, Reihenfolge der Trainingsbilder und
    Augmentation. Die Aufteilung in Training und Validierung bleibt fuer alle
    Seeds gleich (config.SEED), damit sich Laeufe nur im Zufall des Trainings
    unterscheiden. Rueckgabe: Modell (beste Gewichte), Trainingsverlauf, Zusammenfassung.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Trainingsdaten nicht gefunden: {data_dir}. Bitte 03_preprocessing.py ausfuehren."
        )

    # Fester Seed fuer Python, NumPy und TensorFlow sowie deterministische Operationen
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    common = dict(
        validation_split=config.VALIDATION_SPLIT,
        seed=config.SEED,  # gleiche Aufteilung fuer alle Laeufe
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
        verbose=bool(verbose),
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(data_dir, subset="validation", **common)
    class_names = train_ds.class_names
    if class_names != config.CLASSES:
        raise ValueError(f"Unerwartete Klassen: {class_names}")

    counts, weights = class_weights_for(data_dir, class_names)
    class_weight = weights if config.USE_CLASS_WEIGHTS else None

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    model = build_model(num_classes=len(class_names), augment=augment)
    if verbose:
        model.summary()

    # Beide Callbacks ueberwachen val_loss: die gespeicherte Datei ist genau die beste Epoche
    model_path.parent.mkdir(parents=True, exist_ok=True)
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=verbose,
        ),
        callbacks.ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True, verbose=verbose),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb,
        class_weight=class_weight,
        verbose=verbose,
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.index = pd.RangeIndex(1, len(hist_df) + 1, name="epoch")
    best_epoch = int(np.argmin(hist_df["val_loss"].to_numpy())) + 1
    summary = {
        "epochs_run": int(len(hist_df)),
        "best_epoch": best_epoch,
        "best_val_loss": float(hist_df.loc[best_epoch, "val_loss"]),
        "val_accuracy_at_best_epoch": float(hist_df.loc[best_epoch, "val_accuracy"]),
        "train_accuracy_at_best_epoch": float(hist_df.loc[best_epoch, "accuracy"]),
        "training_images_per_class": counts,  # inkl. 20 % Validierung
        "class_weights": {c: round(weights[i], 4) for i, c in enumerate(class_names)}
        if class_weight
        else None,
        "augmentation": dict(config.AUGMENTATION) if augment else None,
        "seed": seed,
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
    }
    return model, hist_df, summary
