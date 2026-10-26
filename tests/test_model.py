"""Modellarchitektur und Augmentation."""

import numpy as np
import tensorflow as tf

from cerebrocam.model import build_model


def test_architecture_matches_project_week_model():
    model = build_model()
    assert model.output_shape == (None, 4)
    assert model.count_params() == 5_402_308  # wie im Original aus der Projektwoche
    names = [layer.name for layer in model.layers]
    assert {"conv1", "conv2", "conv3", "output"} <= set(names)


def test_augmentation_adds_layers_but_no_weights():
    plain, augmented = build_model(), build_model(augment=True)
    aug_names = [layer.name for layer in augmented.layers if layer.name.startswith("aug_")]
    assert aug_names == ["aug_flip", "aug_rotation", "aug_translation", "aug_zoom"]
    assert augmented.count_params() == plain.count_params()


def test_augmentation_is_inactive_at_inference():
    tf.keras.utils.set_random_seed(0)
    augmented = build_model(augment=True)
    plain = build_model(augment=False)
    plain.set_weights(augmented.get_weights())
    x = np.random.default_rng(0).uniform(0, 255, size=(4, 150, 150, 3)).astype("float32")
    np.testing.assert_allclose(augmented.predict(x, verbose=0), plain.predict(x, verbose=0), atol=1e-6)
