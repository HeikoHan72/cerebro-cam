"""Gemeinsame Einstellungen und Hilfsfunktionen fuer die Tests."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from cerebrocam import config  # noqa: E402


def synthetic_mri(seed: int, size: int = 256) -> Image.Image:
    """Kuenstliches 'MRT-aehnliches' Graustufenbild: Kopf-Ellipse mit hellen Flecken.

    Die Struktur haengt vom Seed ab; dadurch sind verschiedene Seeds verschiedene
    Bilder, waehrend Kopien desselben Bildes nach Skalierung aehnlich bleiben.
    """
    rng = np.random.default_rng(seed)
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    margin = int(size * rng.uniform(0.08, 0.2))
    draw.ellipse([margin, margin * 0.8, size - margin, size - margin * 0.8], fill=int(rng.integers(80, 140)))
    for _ in range(6):
        x, y = rng.integers(size // 4, 3 * size // 4, size=2)
        r = int(rng.integers(size // 20, size // 6))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=int(rng.integers(150, 255)))
    return img.filter(ImageFilter.GaussianBlur(2))


@pytest.fixture
def tiny_dataset(tmp_path):
    """Kleiner Trainingsordner mit 8 Bildern pro Klasse (Ordnernamen wie config.CLASSES)."""
    root = tmp_path / "train"
    for c_idx, cls in enumerate(config.CLASSES):
        (root / cls).mkdir(parents=True)
        for i in range(8):
            img = synthetic_mri(seed=100 * c_idx + i, size=config.IMG_SIZE[0]).convert("RGB")
            img.save(root / cls / f"{cls}_{i:02d}.jpg", quality=95)
    return root
