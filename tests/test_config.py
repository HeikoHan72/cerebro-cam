"""Plausibilitaet der zentralen Konfiguration."""

from cerebrocam import config


def test_classes_are_sorted_like_keras_reads_them():
    assert config.CLASSES == sorted(config.CLASSES)
    assert len(config.CLASSES) == 4


def test_sartaj_label_map_covers_all_classes():
    assert set(config.SARTAJ_LABEL_MAP.values()) == set(config.CLASSES)
    sartaj_folders = set(config.DATASETS["sartaj"]["expected_counts"]["Training"])
    assert set(config.SARTAJ_LABEL_MAP) == sartaj_folders


def test_expected_image_counts():
    nick = config.DATASETS["nickparvar"]["expected_counts"]
    sartaj = config.DATASETS["sartaj"]["expected_counts"]
    assert sum(nick["Training"].values()) == 5600
    assert sum(nick["Testing"].values()) == 1600
    assert sum(sartaj["Training"].values()) + sum(sartaj["Testing"].values()) == 3264


def test_dataset_markers_are_distinct():
    markers = [info["marker"] for info in config.DATASETS.values()]
    assert len(set(markers)) == len(markers)


def test_paths_stay_inside_project():
    for name in ("DATA_DIR", "MODELS_DIR", "RESULTS_DIR", "FIGURES_DIR", "DUPLICATES_DIR"):
        assert getattr(config, name).is_relative_to(config.PROJECT_ROOT)


def test_duplicate_thresholds_are_ordered():
    assert 0 <= config.DUPLICATE_MAX_DISTANCE < config.GRAY_ZONE_MAX_DISTANCE < 64
