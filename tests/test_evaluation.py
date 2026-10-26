"""Kennzahlen und Zuordnung von Vorhersagen zu Manifest und Mapping."""

import pandas as pd
import pytest

from cerebrocam import config
from cerebrocam.evaluation import (
    compute_metrics,
    external_subsets,
    internal_subsets,
    label_external_predictions,
    label_internal_predictions,
    metrics_by_subset,
)


def test_compute_metrics_on_toy_example():
    m = compute_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"], labels=["a", "b"])
    assert m["n"] == 4
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["per_class"]["a"]["recall"] == pytest.approx(0.5)
    assert m["per_class"]["b"]["precision"] == pytest.approx(2 / 3)


@pytest.fixture
def manifest_csv(tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "split": ["Training", "Testing", "Testing"],
            "class": ["glioma", "glioma", "glioma"],
            "file": [
                "Training/glioma/Tr-gl_1.jpg",
                "Testing/glioma/Te-gl_1.jpg",
                "Testing/glioma/Te-gl_2.jpg",
            ],
            "keep": [True, True, False],
            "reason": ["kept", "kept", "duplicate_of_training_image"],
        }
    ).to_csv(path, index=False)
    return path


def test_label_internal_predictions(manifest_csv):
    preds = pd.DataFrame(
        {
            "file": ["glioma/Te-gl_1.jpg", "glioma/Te-gl_2.jpg"],
            "predicted_class": ["glioma", "notumor"],
            "confidence": [0.9, 0.8],
        }
    )
    df = label_internal_predictions(preds, manifest_csv)
    assert df["true_class"].tolist() == ["glioma", "glioma"]
    assert df["keep"].tolist() == [True, False]
    subsets = internal_subsets(df)
    assert subsets["official_test"].sum() == 2 and subsets["deduplicated_test"].sum() == 1
    metrics = metrics_by_subset(df, subsets, config.CLASSES)
    assert metrics["official_test"]["accuracy"] == pytest.approx(0.5)
    assert metrics["deduplicated_test"]["accuracy"] == pytest.approx(1.0)


def test_label_internal_predictions_rejects_unknown_files(manifest_csv):
    preds = pd.DataFrame({"file": ["glioma/unknown.jpg"], "predicted_class": ["glioma"], "confidence": [1.0]})
    with pytest.raises(ValueError):
        label_internal_predictions(preds, manifest_csv)


def test_label_external_predictions(tmp_path):
    mapping = tmp_path / "mapping.csv"
    overlap = tmp_path / "overlap.csv"
    pd.DataFrame(
        {
            "new_filename": ["00001.jpg", "00002.jpg"],
            "true_class": ["no_tumor", "glioma_tumor"],
            "original_path": ["Testing/no_tumor/a.jpg", "Training/glioma_tumor/b.jpg"],
        }
    ).to_csv(mapping, index=False)
    pd.DataFrame(
        {
            "original_path": ["Testing/no_tumor/a.jpg", "Training/glioma_tumor/b.jpg"],
            "category": ["identical", "new"],
            "eval_not_in_training": [False, True],
        }
    ).to_csv(overlap, index=False)
    preds = pd.DataFrame(
        {
            "new_filename": ["00002.jpg", "00001.jpg"],
            "predicted_class": ["glioma", "notumor"],
            "confidence": [1, 1],
        }
    )
    df = label_external_predictions(preds, mapping, overlap, config.SARTAJ_LABEL_MAP)
    assert df.set_index("new_filename").loc["00001.jpg", "true_class"] == "notumor"
    subsets = external_subsets(df)
    assert subsets["all"].sum() == 2 and subsets["not_in_training"].sum() == 1

    with pytest.raises(ValueError):
        label_external_predictions(preds.iloc[:1], mapping, overlap, config.SARTAJ_LABEL_MAP)
