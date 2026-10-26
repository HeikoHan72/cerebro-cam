"""Vorverarbeitung: Farbmodus, Groesse und Format der Ausgabebilder."""

import pytest
from PIL import Image

from cerebrocam.preprocessing import preprocess_image


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P"])
def test_output_is_rgb_jpeg_of_target_size(tmp_path, mode):
    src = tmp_path / f"in_{mode}.png"
    Image.new(mode, (300, 200)).save(src)
    dest = tmp_path / "out" / f"{mode}.jpg"
    preprocess_image(src, dest)
    with Image.open(dest) as img:
        assert img.format == "JPEG"
        assert img.mode == "RGB"
        assert img.size == (150, 150)


def test_non_square_size_uses_height_width_order(tmp_path):
    src = tmp_path / "in.png"
    Image.new("L", (64, 64)).save(src)
    dest = tmp_path / "out.jpg"
    preprocess_image(src, dest, img_size=(100, 200))  # (Hoehe, Breite)
    with Image.open(dest) as img:
        assert img.size == (200, 100)  # PIL: (Breite, Hoehe)
