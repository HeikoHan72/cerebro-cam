"""Aufbau des CerebroCAM-CNN (unveraenderte Architektur aus der Projektwoche)."""

from tensorflow import keras
from tensorflow.keras import layers

from cerebrocam import config


def build_model(num_classes: int = len(config.CLASSES), img_size: tuple = config.IMG_SIZE) -> keras.Model:
    """Erstellt und kompiliert das CNN: drei Faltungsbloecke und ein Dense-Klassifikator.

    Die Schichten tragen feste Namen, damit Grad-CAM die letzte Faltungsschicht
    ('conv3') eindeutig findet.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(*img_size, 3)),
            layers.Rescaling(1.0 / 255, name="rescaling"),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1"),
            layers.MaxPooling2D(2, 2, name="pool1"),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"),
            layers.MaxPooling2D(2, 2, name="pool2"),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3"),
            layers.MaxPooling2D(2, 2, name="pool3"),
            layers.Flatten(name="flatten"),
            layers.Dense(128, activation="relu", name="dense"),
            layers.Dropout(0.5, name="dropout"),
            layers.Dense(num_classes, activation="softmax", name="output"),
        ],
        name="cerebrocam_cnn",
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
