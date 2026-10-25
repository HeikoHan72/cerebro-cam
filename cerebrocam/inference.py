"""Vorhersagen fuer alle Bilder eines Ordners, eingelesen wie im Training."""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from cerebrocam import config


def predict_folder(
    model: tf.keras.Model, folder: Path, class_names: list[str] = config.CLASSES
) -> pd.DataFrame:
    """Klassifiziert alle Bilder in folder (auch in Unterordnern).

    Die Bilder werden mit image_dataset_from_directory geladen, also genauso wie
    beim Training. Rueckgabe: Dateipfad relativ zu folder, vorhergesagte Klasse,
    Konfidenz (hoechste Softmax-Wahrscheinlichkeit).
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        folder,
        labels=None,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        verbose=False,
    )
    probabilities = model.predict(dataset, verbose=0)
    return pd.DataFrame(
        {
            "file": [Path(p).relative_to(folder).as_posix() for p in dataset.file_paths],
            "predicted_class": [class_names[i] for i in np.argmax(probabilities, axis=1)],
            "confidence": probabilities.max(axis=1).round(4),
        }
    )
