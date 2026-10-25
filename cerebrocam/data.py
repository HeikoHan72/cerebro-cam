"""Hilfsfunktionen zum Einrichten und Pruefen der Datensaetze."""

import hashlib
import zipfile
from pathlib import Path

from cerebrocam import config


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Berechnet die SHA256-Pruefsumme einer Datei."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identify_zip(zip_path: Path, datasets: dict = config.DATASETS) -> str | None:
    """Erkennt anhand der Ordnerstruktur, zu welchem Datensatz eine ZIP gehoert.

    Beide Kaggle-Downloads heissen 'archive.zip'; unterscheiden lassen sie sich
    nur am Inhalt (z. B. 'Training/glioma/' gegenueber 'Training/glioma_tumor/').
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    for name, info in datasets.items():
        if any(n.startswith(info["marker"]) for n in names):
            return name
    return None


def count_images(root: Path, extensions: set = config.IMAGE_EXTENSIONS) -> dict:
    """Zaehlt Bilder pro Split und Klasse: {split: {klasse: anzahl}}."""
    counts: dict = {}
    if not root.exists():
        return counts
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        counts[split_dir.name] = {
            cls_dir.name: sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in extensions)
            for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir())
        }
    return counts


def counts_match(actual: dict, expected: dict) -> bool:
    """Prueft, ob die gezaehlten Bilder exakt den erwarteten Zahlen entsprechen."""
    return actual == expected


def list_images(
    folder: Path, recursive: bool = False, extensions: set = config.IMAGE_EXTENSIONS
) -> list[Path]:
    """Liefert alle Bilddateien eines Ordners in fester, sortierter Reihenfolge.

    Die Sortierung ist wichtig: Die Reihenfolge, in der das Dateisystem Dateien
    liefert, kann sich zwischen Rechnern unterscheiden.
    """
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in extensions)
