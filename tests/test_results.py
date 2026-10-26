"""Die committeten Ergebnisse sind in sich stimmig und entsprechen den Zahlen im README.

Diese Tests brauchen keine Bilddaten: Sie rechnen die Kennzahlen aus den
gespeicherten Vorhersagen nach und vergleichen sie mit den Referenzwerten.
"""

import json

import pandas as pd
import pytest

from cerebrocam import config
from cerebrocam.evaluation import compute_metrics, internal_subsets


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_internal_metrics_can_be_recomputed_from_predictions():
    predictions = pd.read_csv(config.PREDICTIONS_INTERNAL_CSV)
    metrics = load_json(config.METRICS_INTERNAL_JSON)
    for name, mask in internal_subsets(predictions).items():
        sub = predictions[mask]
        recomputed = compute_metrics(sub["true_class"], sub["predicted_class"], config.CLASSES)
        assert recomputed["confusion_matrix"] == metrics[name]["confusion_matrix"]
        assert recomputed["accuracy"] == pytest.approx(metrics[name]["accuracy"])


# Referenzwerte, wie sie im README stehen: (Datei, Pfad im JSON, Anzahl Bilder, Accuracy)
REFERENCE_VALUES = [
    # neu trainiertes Modell (Seed 42)
    (config.METRICS_INTERNAL_JSON, ["official_test"], 1600, 0.8800),
    (config.METRICS_INTERNAL_JSON, ["deduplicated_test"], 1112, 0.8606),
    (config.METRICS_EXTERNAL_JSON, ["all"], 3264, 0.8876),
    (config.METRICS_EXTERNAL_JSON, ["not_in_training"], 806, 0.6948),
    # Originalmodell aus der Projektwoche
    (config.METRICS_ORIGINAL_MODEL_JSON, ["internal", "official_test"], 1600, 0.8862),
    (config.METRICS_ORIGINAL_MODEL_JSON, ["internal", "deduplicated_test"], 1112, 0.8606),
    (config.METRICS_ORIGINAL_MODEL_JSON, ["external", "all"], 3264, 0.9059),
    (config.METRICS_ORIGINAL_MODEL_JSON, ["external", "not_in_training"], 806, 0.7134),
]


@pytest.mark.parametrize(("path", "keys", "n", "accuracy"), REFERENCE_VALUES)
def test_reported_reference_values(path, keys, n, accuracy):
    metrics = load_json(path)
    for key in keys:
        metrics = metrics[key]
    assert metrics["n"] == n
    assert round(metrics["accuracy"], 4) == accuracy


def test_confusion_matrices_match_sample_sizes():
    for path in (config.METRICS_INTERNAL_JSON, config.METRICS_EXTERNAL_JSON):
        for name, m in load_json(path).items():
            if not isinstance(m, dict) or "confusion_matrix" not in m:
                continue
            total = sum(map(sum, m["confusion_matrix"]))
            correct = sum(m["confusion_matrix"][i][i] for i in range(len(m["labels"])))
            assert total == m["n"], name
            assert correct / total == pytest.approx(m["accuracy"]), name


def test_duplicate_summary_matches_manifest():
    summary = load_json(config.DUPLICATES_SUMMARY_JSON)
    manifest = pd.read_csv(config.NICKPARVAR_MANIFEST_CSV)
    overlap = pd.read_csv(config.EXTERNAL_OVERLAP_CSV)
    train, test = manifest[manifest["split"] == "Training"], manifest[manifest["split"] == "Testing"]
    assert (len(train), int(train["keep"].sum())) == (5600, 4836)
    assert (len(test), int(test["keep"].sum())) == (1600, 1112)
    assert sum(summary["training"]["kept"].values()) == 4836
    assert sum(summary["test"]["kept"].values()) == 1112
    assert int((overlap["category"] == "identical").sum()) == 2405
    assert int(overlap["eval_not_in_training"].sum()) == 806


def test_duplicate_details_quoted_in_readme():
    manifest = pd.read_csv(config.NICKPARVAR_MANIFEST_CSV)
    summary = load_json(config.DUPLICATES_SUMMARY_JSON)
    reasons = manifest["reason"].value_counts()
    assert reasons["duplicate_within_training"] == 764
    assert reasons["duplicate_of_training_image"] == 439
    assert reasons["duplicate_within_test"] == 49
    kept_test = manifest[(manifest["split"] == "Testing") & manifest["keep"]]
    # Grauzone: Abstand 3-8 zum naechsten Trainingsbild (meist benachbarte Schicht)
    gray_zone = kept_test["nearest_train_distance"].between(
        config.DUPLICATE_MAX_DISTANCE + 1, config.GRAY_ZONE_MAX_DISTANCE
    )
    assert int(gray_zone.sum()) == 569
    assert summary["external"]["byte_identical_to_training"] == 2257


def test_augmentation_summary_matches_runs():
    runs = pd.read_csv(config.AUGMENTATION_RUNS_CSV)
    summary = load_json(config.AUGMENTATION_SUMMARY_JSON)
    metric = summary["decision_metric"]
    assert len(runs) == 2 * len(summary["seeds"])
    assert runs.loc[~runs["augmentation"], metric].mean() == pytest.approx(summary["mean_without"])
    assert runs.loc[runs["augmentation"], metric].mean() == pytest.approx(summary["mean_with"])
    assert summary["adopt_augmentation"] is False
    # Der Lauf mit Seed 42 ohne Augmentation entspricht dem veroeffentlichten Modell
    seed42 = runs[(runs["seed"] == 42) & ~runs["augmentation"]].iloc[0]
    assert round(seed42["dedup_accuracy"], 4) == round(
        load_json(config.METRICS_INTERNAL_JSON)["deduplicated_test"]["accuracy"], 4
    )
