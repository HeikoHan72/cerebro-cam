"""Duplikat-Erkennung: Fingerabdruecke, Distanzen, Gruppen."""

import numpy as np
import pandas as pd

from cerebrocam import config
from cerebrocam.duplicates import (
    distance_matrix,
    duplicate_groups,
    fingerprint_images,
    image_fingerprint,
    nearest_neighbors,
)
from tests.conftest import synthetic_mri


def hashes(values):
    """DataFrame mit identischem pHash und dHash, damit Distanzen leicht nachzurechnen sind."""
    arr = np.array(values, dtype=np.uint64)
    return pd.DataFrame({"phash": arr, "dhash": arr})


def test_resized_and_recompressed_copy_counts_as_duplicate(tmp_path):
    original = tmp_path / "original.png"
    copy = tmp_path / "copy.jpg"
    img = synthetic_mri(seed=1)
    img.save(original)
    img.resize((150, 150), resample=3).save(copy, quality=75)
    df = fingerprint_images([original, copy], tmp_path)
    assert distance_matrix(df.iloc[[0]], df.iloc[[1]])[0, 0] <= config.DUPLICATE_MAX_DISTANCE


def test_different_images_are_far_apart(tmp_path):
    paths = []
    for seed in (1, 2, 3):
        p = tmp_path / f"img_{seed}.png"
        synthetic_mri(seed=seed).save(p)
        paths.append(p)
    df = fingerprint_images(paths, tmp_path)
    d = distance_matrix(df, df)
    off_diagonal = d[~np.eye(len(df), dtype=bool)]
    assert off_diagonal.min() > config.GRAY_ZONE_MAX_DISTANCE


def test_fingerprint_contains_md5_and_hashes(tmp_path):
    p = tmp_path / "img.png"
    synthetic_mri(seed=5).save(p)
    fp = image_fingerprint(p)
    assert len(fp["md5"]) == 32
    assert 0 <= fp["phash"] < 2**64 and 0 <= fp["dhash"] < 2**64


def test_distance_is_max_of_both_hamming_distances():
    a = pd.DataFrame({"phash": np.array([0b1], dtype=np.uint64), "dhash": np.array([0b0], dtype=np.uint64)})
    b = pd.DataFrame({"phash": np.array([0b0], dtype=np.uint64), "dhash": np.array([0b111], dtype=np.uint64)})
    assert distance_matrix(a, b)[0, 0] == 3


def test_groups_follow_chains_and_use_smallest_index():
    # 0 und 1 haben Distanz 2, 1 und 2 haben Distanz 2, 0 und 2 haben Distanz 4; 3 ist weit weg
    df = hashes([0b0, 0b11, 0b1111, 2**64 - 1])
    assert duplicate_groups(df, max_distance=2).tolist() == [0, 0, 0, 3]
    assert duplicate_groups(df, max_distance=1).tolist() == [0, 1, 2, 3]


def test_nearest_neighbors():
    reference = hashes([0b0, 0b1111_0000, 2**64 - 1])
    query = hashes([0b1, 0b1110_0000])
    idx, dist = nearest_neighbors(query, reference, chunk=1)
    assert idx.tolist() == [0, 1]
    assert dist.tolist() == [1, 1]
