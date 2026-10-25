"""Erkennung von identischen und nahezu identischen Bildern per Bild-Fingerabdruck.

Jedes Bild bekommt zwei 64-Bit-Fingerabdruecke (pHash und dHash) aus seiner
Graustufenversion. Die Distanz zweier Bilder ist das Maximum der beiden
Hamming-Distanzen: Beide Verfahren muessen also uebereinstimmen. Verkleinerte,
neu komprimierte oder leicht im Kontrast veraenderte Kopien haben eine kleine
Distanz, verschiedene Bilder eine grosse.
"""

import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image


def image_fingerprint(path: Path) -> dict:
    """MD5 der Datei sowie pHash und dHash (als Ganzzahlen) eines Bildes."""
    with Image.open(path) as img:
        gray = img.convert("L")
        phash = int(str(imagehash.phash(gray)), 16)
        dhash = int(str(imagehash.dhash(gray)), 16)
    return {"md5": hashlib.md5(path.read_bytes()).hexdigest(), "phash": phash, "dhash": dhash}


def fingerprint_images(paths: list[Path], root: Path) -> pd.DataFrame:
    """Fingerabdruecke fuer eine Liste von Bildern; 'file' ist der Pfad relativ zu root."""
    rows = [{"file": p.relative_to(root).as_posix(), **image_fingerprint(p)} for p in paths]
    df = pd.DataFrame(rows, columns=["file", "md5", "phash", "dhash"])
    df["phash"] = df["phash"].astype(np.uint64)
    df["dhash"] = df["dhash"].astype(np.uint64)
    return df


def distance_matrix(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """Distanzmatrix max(Hamming(pHash), Hamming(dHash)) zwischen allen Bildern aus a und b."""
    pa, pb = a["phash"].to_numpy(np.uint64), b["phash"].to_numpy(np.uint64)
    da, db = a["dhash"].to_numpy(np.uint64), b["dhash"].to_numpy(np.uint64)
    return np.maximum(
        np.bitwise_count(pa[:, None] ^ pb[None, :]), np.bitwise_count(da[:, None] ^ db[None, :])
    )


def nearest_neighbors(query: pd.DataFrame, reference: pd.DataFrame, chunk: int = 1024) -> tuple:
    """Fuer jedes Bild aus query: Index und Distanz des aehnlichsten Bildes aus reference."""
    idx = np.empty(len(query), dtype=np.int64)
    dist = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), chunk):
        d = distance_matrix(query.iloc[start : start + chunk], reference)
        idx[start : start + chunk] = d.argmin(axis=1)
        dist[start : start + chunk] = d.min(axis=1)
    return idx, dist


def duplicate_groups(df: pd.DataFrame, max_distance: int, chunk: int = 1024) -> np.ndarray:
    """Fasst Bilder mit Distanz <= max_distance zu Gruppen zusammen (auch ueber Ketten).

    Rueckgabe: Gruppennummer je Bild = kleinster Index der Gruppe. Bei sortierter
    Eingabe ist das erste Bild einer Gruppe damit ihr Vertreter.
    """
    n = len(df)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, n, chunk):
        d = distance_matrix(df.iloc[start : start + chunk], df)
        rows, cols = np.nonzero(d <= max_distance)
        for r, c in zip(rows + start, cols, strict=True):
            if c > r:
                ri, rc = find(r), find(c)
                if ri != rc:
                    parent[max(ri, rc)] = min(ri, rc)
    roots = np.array([find(i) for i in range(n)])
    # Gruppennummer = kleinster Index unter den Mitgliedern
    smallest = pd.Series(np.arange(n)).groupby(roots).transform("min").to_numpy()
    return smallest
