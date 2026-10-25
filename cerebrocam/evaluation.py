"""Kennzahlen berechnen, speichern und Konfusionsmatrizen zeichnen."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def compute_metrics(y_true, y_pred, labels: list[str]) -> dict:
    """Accuracy, Precision/Recall/F1 pro Klasse und Konfusionsmatrix als Dictionary."""
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            c: {
                "precision": float(report[c]["precision"]),
                "recall": float(report[c]["recall"]),
                "f1-score": float(report[c]["f1-score"]),
                "support": int(report[c]["support"]),
            }
            for c in labels
        },
        "macro_avg": {k: float(report["macro avg"][k]) for k in ("precision", "recall", "f1-score")},
        "labels": list(labels),
        "confusion_matrix": cm.tolist(),
    }


def metrics_by_subset(df: pd.DataFrame, subsets: dict, labels: list[str]) -> dict:
    """Kennzahlen fuer mehrere Teilmengen; df braucht die Spalten true_class und predicted_class."""
    return {
        name: compute_metrics(df.loc[mask, "true_class"], df.loc[mask, "predicted_class"], labels)
        for name, mask in subsets.items()
    }


def label_internal_predictions(predictions: pd.DataFrame, manifest_csv: Path) -> pd.DataFrame:
    """Ergaenzt Vorhersagen fuer den internen Testsatz um wahre Klasse und Duplikat-Status.

    predictions: Spalten 'file' (z. B. 'glioma/Te-gl_1.jpg'), 'predicted_class', 'confidence'.
    Die wahre Klasse ist der Ordnername; 'keep' und 'reason' stammen aus dem Manifest (Skript 02).
    """
    df = predictions.copy()
    df["true_class"] = df["file"].str.split("/").str[0]
    df["key"] = df["file"].str.rsplit(".", n=1).str[0]

    manifest = pd.read_csv(manifest_csv)
    manifest = manifest[manifest["split"] == "Testing"].copy()
    # 'Testing/glioma/Te-gl_1.jpg' -> 'glioma/Te-gl_1' (vorverarbeitete Bilder enden immer auf .jpg)
    manifest["key"] = manifest["file"].str.split("/", n=1).str[1].str.rsplit(".", n=1).str[0]

    df = df.merge(manifest[["key", "keep", "reason"]], on="key", how="left", validate="one_to_one")
    if df["keep"].isna().any():
        raise ValueError(
            "Nicht alle Testbilder stehen im Manifest. Bitte die Skripte 02 und 03 erneut ausfuehren."
        )
    df["keep"] = df["keep"].astype(bool)
    return df.drop(columns="key")


def internal_subsets(df: pd.DataFrame) -> dict:
    """Offizieller Testsatz (alle Bilder) und bereinigter Testsatz (ohne Duplikate)."""
    return {"official_test": pd.Series(True, index=df.index), "deduplicated_test": df["keep"]}


def label_external_predictions(
    predictions: pd.DataFrame, mapping_csv: Path, overlap_csv: Path, label_map: dict
) -> pd.DataFrame:
    """Entblindet die externen Vorhersagen und ergaenzt die Ueberlappung mit dem Training.

    predictions: Spalten 'new_filename', 'predicted_class', 'confidence' (Skript 07).
    """
    mapping = pd.read_csv(mapping_csv)
    overlap = pd.read_csv(overlap_csv)
    df = predictions.merge(mapping, on="new_filename", how="inner", validate="one_to_one")
    df = df.merge(
        overlap[["original_path", "category", "eval_not_in_training"]],
        on="original_path",
        how="left",
        validate="one_to_one",
    )
    if len(df) != len(mapping) or df["category"].isna().any():
        raise ValueError("Nicht alle externen Bilder konnten zugeordnet werden (Skripte 02, 06, 07 pruefen).")
    df["true_class"] = df["true_class"].map(label_map)
    if df["true_class"].isna().any():
        raise ValueError("Unbekannte Klassennamen in der Mapping-Tabelle.")
    df["eval_not_in_training"] = df["eval_not_in_training"].astype(bool)
    return df


def external_subsets(df: pd.DataFrame) -> dict:
    """Alle externen Bilder und nur die, die nicht (nahezu) identisch im Training vorkommen."""
    return {"all": pd.Series(True, index=df.index), "not_in_training": df["eval_not_in_training"]}


def save_json(obj: dict, path: Path) -> None:
    """Speichert ein Dictionary als lesbare JSON-Datei."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_confusion_matrix(cm, labels: list[str], title: str, path: Path, cmap: str = "Blues") -> None:
    """Zeichnet eine Konfusionsmatrix als Heatmap und speichert sie als PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.asarray(cm), annot=True, fmt="d", cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()
