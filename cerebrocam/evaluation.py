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
