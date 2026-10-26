"""Aufbau des CerebroCAM-CNN (unveraenderte Architektur aus der Projektwoche)."""

from tensorflow import keras
from tensorflow.keras import layers

from cerebrocam import config


def augmentation_layers(params: dict = config.AUGMENTATION) -> list:
    """Zufaellige Bildveraenderungen, die Keras nur im Training anwendet.

    Freie Flaechen nach Drehung, Verschiebung oder Zoom werden schwarz gefuellt,
    passend zum schwarzen Hintergrund der MRT-Bilder.
    """
    fill = {"fill_mode": "constant", "fill_value": 0.0}
    result = []
    if params.get("horizontal_flip"):
        result.append(layers.RandomFlip("horizontal", name="aug_flip"))
    if params.get("rotation_degrees"):
        result.append(layers.RandomRotation(params["rotation_degrees"] / 360, name="aug_rotation", **fill))
    if params.get("translation"):
        t = params["translation"]
        result.append(layers.RandomTranslation(t, t, name="aug_translation", **fill))
    if params.get("zoom"):
        result.append(layers.RandomZoom(params["zoom"], name="aug_zoom", **fill))
    return result


def build_model(
    num_classes: int = len(config.CLASSES), img_size: tuple = config.IMG_SIZE, augment: bool = False
) -> keras.Model:
    """Erstellt und kompiliert das CNN: drei Faltungsbloecke und ein Dense-Klassifikator.

    Mit augment=True werden vor der Normierung Augmentation-Schichten eingefuegt;
    die eigentliche Architektur bleibt unveraendert. Die Schichten tragen feste
    Namen, damit Grad-CAM die letzte Faltungsschicht ('conv3') eindeutig findet.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(*img_size, 3)),
            *(augmentation_layers() if augment else []),
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
