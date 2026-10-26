"""Grad-CAM: zeigt, welche Bildbereiche eine Vorhersage des CNN beeinflusst haben.

Verfahren (Selvaraju et al., 2017): Man berechnet den Gradienten des Klassen-Scores
(Logit vor Softmax) nach den Feature-Maps der letzten Faltungsschicht, mittelt ihn
pro Kanal zu einem Gewicht und bildet die gewichtete Summe der Feature-Maps.
Positive Werte markieren Bereiche, die fuer die Klasse sprechen.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from cerebrocam import config


def build_gradcam_model(model: keras.Model, layer_name: str = "conv3") -> keras.Model:
    """Hilfsmodell: Eingabebild -> (Feature-Maps von layer_name, Logits der Ausgabeschicht).

    Die Schichten des trainierten Modells werden wiederverwendet (gleiche Gewichte).
    Die Logits werden aus dem Eingang der Ausgabeschicht berechnet, weil Gradienten
    hinter einer Softmax bei sehr sicheren Vorhersagen nahezu verschwinden.
    """
    inputs = keras.Input(shape=model.input_shape[1:])
    x, feature_maps = inputs, None
    output_layer = model.get_layer("output")
    for layer in model.layers:
        if layer is output_layer:
            break
        x = layer(x)  # Augmentation und Dropout sind ausserhalb des Trainings inaktiv
        if layer.name == layer_name:
            feature_maps = x
    if feature_maps is None:
        raise ValueError(f"Schicht '{layer_name}' nicht im Modell gefunden.")
    logits = keras.ops.add(keras.ops.matmul(x, output_layer.kernel), output_layer.bias)
    return keras.Model(inputs, [feature_maps, logits])


def load_image(path: Path, img_size: tuple = config.IMG_SIZE) -> np.ndarray:
    """Laedt ein Bild genau wie image_dataset_from_directory (TensorFlow-Dekodierung, bilinear).

    Rueckgabe: float32-Array (Hoehe, Breite, 3) mit Werten 0-255.
    """
    raw = tf.io.read_file(str(path))
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size, method="bilinear")
    return img.numpy().astype(np.float32)


def gradcam(grad_model: keras.Model, image: np.ndarray, class_index: int | None = None) -> tuple:
    """Grad-CAM-Karte fuer ein Bild.

    Rueckgabe: (Karte in Bildgroesse mit Werten 0-1, Klassenindex, Softmax-Wahrscheinlichkeiten).
    Ohne class_index wird die vorhergesagte Klasse erklaert.
    """
    batch = tf.convert_to_tensor(image[None, ...])
    with tf.GradientTape() as tape:
        feature_maps, logits = grad_model(batch, training=False)
        if class_index is None:
            class_index = int(tf.argmax(logits[0]))
        score = logits[:, class_index]
    grads = tape.gradient(score, feature_maps)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # ein Gewicht pro Kanal
    cam = tf.nn.relu(tf.reduce_sum(feature_maps[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], image.shape[:2], method="bilinear")[..., 0]
    probabilities = tf.nn.softmax(logits[0]).numpy()
    return cam.numpy(), class_index, probabilities
