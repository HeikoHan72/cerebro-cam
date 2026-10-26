#!/usr/bin/env python3
"""
SKRIPT 11: EXPERIMENT DATA AUGMENTATION (mit / ohne, je 3 Seeds)

Trainiert das CNN je dreimal ohne und mit Augmentation (Seeds aus
config.EXPERIMENT_SEEDS) und bewertet jedes Modell auf dem bereinigten
Testsatz und extern. Ein einzelner Lauf pro Variante waere nicht
aussagekraeftig, weil schon der Zufall des Trainings die Accuracy um rund
einen Prozentpunkt verschiebt.

Entscheidungsregel (vorab festgelegt): Augmentation wird uebernommen, wenn der
Mittelwert der Validierungs-Accuracy (beste Epoche) mit Augmentation um mehr als
die groessere der beiden Standardabweichungen ueber dem Mittelwert ohne liegt.
Entschieden wird bewusst auf der Validierung, nicht auf dem Testsatz: Sonst
wuerde das Modell indirekt auf den Testsatz optimiert. Die Test-Kennzahlen
werden fuer beide Varianten nur berichtet.

Laufzeit: etwa 1 bis 1,5 Stunden auf einer CPU (mit Augmentation lernt das
Modell langsamer und braucht mehr Epochen). Jeder abgeschlossene Lauf wird sofort
gespeichert; nach einer Unterbrechung setzt das Skript beim naechsten Lauf fort.
Voraussetzung: Skripte 02, 03 und 06 wurden ausgefuehrt.
"""

# %% ============================================================================
# BLOCK 1: IMPORTE & KONFIGURATION
# ==============================================================================

import time

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

from cerebrocam import config
from cerebrocam.evaluation import (
    external_subsets,
    internal_subsets,
    label_external_predictions,
    label_internal_predictions,
    metrics_by_subset,
    save_json,
)
from cerebrocam.inference import predict_folder
from cerebrocam.training import train_model

RUNS = [(seed, augment) for augment in (False, True) for seed in config.EXPERIMENT_SEEDS]
HAS_EXTERNAL = config.EXTERNAL_UNLABELED_DIR.exists() and config.EXTERNAL_MAPPING_CSV.exists()
config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(model) -> dict:
    """Kennzahlen eines Modells: interner Test (offiziell/bereinigt) und extern."""
    internal = label_internal_predictions(
        predict_folder(model, config.TEST_PROCESSED_DIR), config.NICKPARVAR_MANIFEST_CSV
    )
    m = metrics_by_subset(internal, internal_subsets(internal), config.CLASSES)
    dedup = m["deduplicated_test"]
    row = {
        "official_accuracy": m["official_test"]["accuracy"],
        "dedup_accuracy": dedup["accuracy"],
        "dedup_macro_f1": dedup["macro_avg"]["f1-score"],
        **{f"dedup_recall_{c}": dedup["per_class"][c]["recall"] for c in config.CLASSES},
        "dedup_precision_notumor": dedup["per_class"]["notumor"]["precision"],
    }
    if HAS_EXTERNAL:
        preds = predict_folder(model, config.EXTERNAL_UNLABELED_DIR).rename(columns={"file": "new_filename"})
        ext = label_external_predictions(
            preds, config.EXTERNAL_MAPPING_CSV, config.EXTERNAL_OVERLAP_CSV, config.SARTAJ_LABEL_MAP
        )
        e = metrics_by_subset(ext, external_subsets(ext), config.CLASSES)
        row["external_all_accuracy"] = e["all"]["accuracy"]
        row["external_not_in_training_accuracy"] = e["not_in_training"]["accuracy"]
    return row


# %% ============================================================================
# BLOCK 2: LAEUFE (bereits erledigte werden uebersprungen)
# ==============================================================================

done = pd.read_csv(config.AUGMENTATION_RUNS_CSV) if config.AUGMENTATION_RUNS_CSV.exists() else pd.DataFrame()
for i, (seed, augment) in enumerate(RUNS, start=1):
    if not done.empty and ((done["seed"] == seed) & (done["augmentation"] == augment)).any():
        print(f"[{i}/{len(RUNS)}] Seed {seed}, Augmentation {augment}: bereits vorhanden, uebersprungen")
        continue
    print(f"\n[{i}/{len(RUNS)}] Training mit Seed {seed}, Augmentation {augment} ...")
    start = time.time()
    keras.backend.clear_session()
    model_path = config.EXPERIMENT_MODELS_DIR / f"aug_{str(augment).lower()}_seed{seed}.keras"
    model, hist_df, summary = train_model(seed=seed, augment=augment, model_path=model_path, verbose=0)
    row = {
        "seed": seed,
        "augmentation": augment,
        "epochs_run": summary["epochs_run"],
        "best_epoch": summary["best_epoch"],
        "best_val_loss": summary["best_val_loss"],
        "val_accuracy": summary["val_accuracy_at_best_epoch"],
        **evaluate(model),
        "minutes": round((time.time() - start) / 60, 1),
    }
    done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
    done.to_csv(config.AUGMENTATION_RUNS_CSV, index=False)
    print(
        f"    beste Epoche {row['best_epoch']} von {row['epochs_run']}, "
        f"bereinigte Test-Accuracy {row['dedup_accuracy']:.4f} ({row['minutes']} min)"
    )

# %% ============================================================================
# BLOCK 3: AUSWERTUNG & ENTSCHEIDUNG
# ==============================================================================

runs = pd.read_csv(config.AUGMENTATION_RUNS_CSV)
metrics = [
    c for c in runs.columns if c.endswith("accuracy") or c.endswith("f1") or "recall" in c or "precision" in c
]
stats = runs.groupby("augmentation")[metrics].agg(["mean", "std", "min", "max"])

DECISION_METRIC = "val_accuracy"  # Validierung, nicht Testsatz
base = runs.loc[~runs["augmentation"], DECISION_METRIC]
aug = runs.loc[runs["augmentation"], DECISION_METRIC]
difference = aug.mean() - base.mean()
spread = max(base.std(ddof=1), aug.std(ddof=1))
adopt = bool(difference > spread)

summary = {
    "seeds": config.EXPERIMENT_SEEDS,
    "augmentation_params": config.AUGMENTATION,
    "max_epochs": config.EPOCHS,
    "decision_metric": DECISION_METRIC,
    "mean_without": float(base.mean()),
    "mean_with": float(aug.mean()),
    "std_without": float(base.std(ddof=1)),
    "std_with": float(aug.std(ddof=1)),
    "difference": float(difference),
    "spread_threshold": float(spread),
    "adopt_augmentation": adopt,
    "per_variant": {
        ("with" if a else "without"): {
            m: {k: float(stats.loc[a, (m, k)]) for k in ("mean", "std", "min", "max")} for m in metrics
        }
        for a in stats.index
    },
}
save_json(summary, config.AUGMENTATION_SUMMARY_JSON)


def fmt(values: pd.Series) -> str:
    """Mittelwert +/- Standardabweichung und Einzelwerte."""
    single = ", ".join(f"{v:.4f}" for v in values)
    return f"{values.mean():.4f} +/- {values.std(ddof=1):.4f}  ({single})"


test_base = runs.loc[~runs["augmentation"], "dedup_accuracy"]
test_aug = runs.loc[runs["augmentation"], "dedup_accuracy"]
print("\nValidierungs-Accuracy, beste Epoche (Mittelwert +/- Standardabweichung ueber die Seeds):")
print(f"  ohne Augmentation: {fmt(base)}")
print(f"  mit Augmentation:  {fmt(aug)}")
print(f"  Differenz: {difference:+.4f}, Schwelle (groessere Streuung): {spread:.4f}")
print(f"Entscheidung: Augmentation {'UEBERNEHMEN' if adopt else 'NICHT uebernehmen'}")
print("\nZur Information - bereinigte Test-Accuracy:")
print(f"  ohne Augmentation: {fmt(test_base)}")
print(f"  mit Augmentation:  {fmt(test_aug)}")

# %% ============================================================================
# BLOCK 4: ABBILDUNG
# Punktdiagramm: jeder Punkt ist ein Seed, der Strich der Mittelwert.
# Links die Entscheidungsgroesse (Validierung), rechts der bereinigte Testsatz.
# ==============================================================================

colors = {False: "#2a78d6", True: "#eb6834"}
markers = {False: "o", True: "s"}
labels = {False: "Without\naugmentation", True: "With\naugmentation"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for ax, column, title in [
    (axes[0], "val_accuracy", "Validation (decision)"),
    (axes[1], "dedup_accuracy", "Deduplicated test set"),
]:
    for x, a in enumerate((False, True)):
        values = runs.loc[runs["augmentation"] == a, column].to_numpy()
        ax.scatter(
            [x] * len(values),
            values * 100,
            s=64,
            color=colors[a],
            marker=markers[a],
            zorder=3,
            edgecolors="white",
            linewidths=1.5,
        )
        mean = values.mean() * 100
        ax.hlines(mean, x - 0.18, x + 0.18, color=colors[a], linewidth=2, zorder=2)
        ax.text(x + 0.22, mean, f"mean {mean:.1f} %", va="center", fontsize=10, color="#52514e")
    ax.set_xticks([0, 1], [labels[False], labels[True]])
    ax.set_xlim(-0.5, 1.9)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("Accuracy (%)")
fig.suptitle(f"Data augmentation: {len(config.EXPERIMENT_SEEDS)} seeds per variant", fontsize=12)
plt.tight_layout()
out = config.FIGURES_DIR / "augmentation_comparison.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300)
plt.show()
plt.close(fig)

print(f"\nLaeufe:          {config.AUGMENTATION_RUNS_CSV}")
print(f"Zusammenfassung: {config.AUGMENTATION_SUMMARY_JSON}")
print(f"Abbildung:       {out}")
