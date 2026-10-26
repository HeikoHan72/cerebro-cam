"""Grad-CAM: Form und Wertebereich der Karte, Konsistenz mit dem Modell."""

import numpy as np
import pytest
import tensorflow as tf

from cerebrocam.gradcam import build_gradcam_model, gradcam, load_image
from cerebrocam.model import build_model
from tests.conftest import synthetic_mri


@pytest.fixture(scope="module")
def model():
    tf.keras.utils.set_random_seed(0)
    return build_model()


def test_gradcam_map_and_probabilities(model):
    grad_model = build_gradcam_model(model)
    image = np.random.default_rng(1).uniform(0, 255, size=(150, 150, 3)).astype("float32")
    cam, class_index, probs = gradcam(grad_model, image)
    assert cam.shape == (150, 150)
    assert cam.min() >= 0 and cam.max() <= 1 + 1e-6
    # Die aus den Logits berechneten Wahrscheinlichkeiten muessen der Modellausgabe entsprechen
    np.testing.assert_allclose(probs, model.predict(image[None], verbose=0)[0], atol=1e-5)
    assert class_index == int(np.argmax(probs))


def test_gradcam_explains_requested_class(model):
    grad_model = build_gradcam_model(model)
    image = np.random.default_rng(2).uniform(0, 255, size=(150, 150, 3)).astype("float32")
    _, class_index, _ = gradcam(grad_model, image, class_index=2)
    assert class_index == 2


def test_unknown_layer_raises(model):
    with pytest.raises(ValueError):
        build_gradcam_model(model, layer_name="does_not_exist")


def test_load_image_matches_keras_dataset_loader(tmp_path):
    folder = tmp_path / "imgs"
    folder.mkdir()
    synthetic_mri(seed=3, size=150).convert("RGB").save(folder / "a.jpg", quality=95)
    dataset = tf.keras.utils.image_dataset_from_directory(
        folder, labels=None, image_size=(150, 150), batch_size=1, shuffle=False, verbose=False
    )
    from_dataset = next(iter(dataset)).numpy()[0]
    np.testing.assert_array_equal(load_image(folder / "a.jpg"), from_dataset)
