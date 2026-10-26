"""Tests mit den echten Kaggle-Daten (werden uebersprungen, wenn die Daten fehlen).

Lokal nach scripts/00_setup_data.py ausfuehrbar; in der CI sind die Daten nicht
vorhanden, weil sie nicht weiterverbreitet werden.
"""

import pandas as pd
import pytest

from cerebrocam import config
from cerebrocam.data import count_images, counts_match, list_images

pytestmark = pytest.mark.skipif(
    not (config.NICKPARVAR_DIR.exists() and config.SARTAJ_DIR.exists()),
    reason="Kaggle-Daten nicht eingerichtet (scripts/00_setup_data.py)",
)


def test_datasets_have_expected_image_counts():
    for info in config.DATASETS.values():
        assert counts_match(count_images(info["target_dir"]), info["expected_counts"])


def test_manifest_covers_exactly_the_nickparvar_images():
    manifest = pd.read_csv(config.NICKPARVAR_MANIFEST_CSV)
    on_disk = {
        p.relative_to(config.NICKPARVAR_DIR).as_posix() for p in list_images(config.NICKPARVAR_DIR, True)
    }
    assert set(manifest["file"]) == on_disk


def test_external_overlap_covers_exactly_the_sartaj_images():
    overlap = pd.read_csv(config.EXTERNAL_OVERLAP_CSV)
    on_disk = {p.relative_to(config.SARTAJ_DIR).as_posix() for p in list_images(config.SARTAJ_DIR, True)}
    assert set(overlap["original_path"]) == on_disk
