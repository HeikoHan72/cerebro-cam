"""Training auf einem winzigen synthetischen Datensatz: laeuft durch und ist reproduzierbar."""

import pandas as pd

from cerebrocam import config
from cerebrocam.training import class_weights_for, train_model


def test_class_weights_are_balanced_for_equal_counts(tiny_dataset):
    counts, weights = class_weights_for(tiny_dataset, config.CLASSES)
    assert counts == {c: 8 for c in config.CLASSES}
    assert all(abs(w - 1.0) < 1e-9 for w in weights.values())


def test_training_runs_and_is_reproducible(tiny_dataset, tmp_path):
    kwargs = dict(seed=7, augment=False, data_dir=tiny_dataset, epochs=2, verbose=0)
    _, hist_a, summary_a = train_model(model_path=tmp_path / "a.keras", **kwargs)
    _, hist_b, _ = train_model(model_path=tmp_path / "b.keras", **kwargs)
    pd.testing.assert_frame_equal(hist_a, hist_b)
    assert (tmp_path / "a.keras").exists()
    assert summary_a["epochs_run"] == 2
    assert 1 <= summary_a["best_epoch"] <= 2
    assert summary_a["seed"] == 7
