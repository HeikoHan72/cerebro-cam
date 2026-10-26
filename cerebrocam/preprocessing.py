"""Bildvorverarbeitung, identisch fuer Trainings-, Test- und externe Bilder."""

from pathlib import Path

from PIL import Image

from cerebrocam import config


def preprocess_image(
    src: Path, dest: Path, img_size: tuple = config.IMG_SIZE, quality: int = config.JPEG_QUALITY
) -> None:
    """Liest ein Bild, wandelt es in RGB, skaliert es mit Lanczos und speichert es als JPEG.

    img_size ist (Hoehe, Breite) wie in Keras; PIL erwartet (Breite, Hoehe).
    """
    with Image.open(src) as img:
        processed = img.convert("RGB").resize((img_size[1], img_size[0]), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    processed.save(dest, "JPEG", quality=quality)
