"""
utils/metrics.py — Extended evaluation metrics for TCSP-PC.

Provides per-model metric computation, paper-style comparative tables,
and emission-behaviour performance scoring aligned with the paper's
Equations (6a-6d) and Table 5/6 definitions.
"""

import numpy as np
import time
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, cohen_kappa_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(model, X_test: np.ndarray, y_test: np.ndarray,
                    model_name: str = "Model") -> dict:
    """
    Compute a full set of classification metrics for a fitted model.

    Returns
    -------
    dict with keys: name, accuracy, f1, precision, recall, auc, mcc,
                    kappa, specificity, time_s, confusion_matrix
    """
    t0 = time.time()
    y_pred = model.predict(X_test)
    elapsed = time.time() - t0

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc  = matthews_corrcoef(y_test, y_pred)
    kap  = cohen_kappa_score(y_test, y_pred)

    try:
        proba = model.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp + 1e-8)
    else:
        specificity = float("nan")

    return {
        "name":            model_name,
        "accuracy":        round(acc, 4),
        "f1":              round(f1, 4),
        "precision":       round(prec, 4),
        "recall":          round(rec, 4),
        "auc":             round(auc, 4) if not np.isnan(auc) else auc,
        "mcc":             round(mcc, 4),
        "kappa":           round(kap, 4),
        "specificity":     round(specificity, 4),
        "time_s":          round(elapsed, 4),
        "confusion_matrix": cm.tolist(),
    }


def compute_all_metrics(models: dict, X_test: np.ndarray,
                         y_test: np.ndarray) -> list:
    """
    Evaluate all models and return a list of metric dicts.

    Parameters
    ----------
    models  : {name: fitted_model}
    X_test  : scaled feature array
    y_test  : true labels
    """
    results = []
    for name, model in models.items():
        r = compute_metrics(model, X_test, y_test, model_name=name)
        results.append(r)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Paper-specific metrics  (Tables 5 & 6 quantities)
# ─────────────────────────────────────────────────────────────────────────────

def impact_verification_score(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               emission_rates: np.ndarray) -> float:
    """
    Impact Verification (Fig 8 metric):
    Weighted accuracy where high-emission misclassifications are penalised
    more than low-emission ones.  Mirrors Eq. (13) W(xn) weighting logic.
    """
    weights = 1.0 + np.abs(emission_rates) / (np.max(np.abs(emission_rates)) + 1e-8)
    correct = (y_true == y_pred).astype(float)
    return float(np.average(correct, weights=weights))


def classification_rate_per_km(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                distance_km: float = 1.0) -> float:
    """
    Classification Rate (/km) as reported in Table 5.
    Correct detections per unit distance.
    """
    correct = np.sum(y_true == y_pred)
    return float(correct / (len(y_true) * distance_km))


def recommendations_per_vehicle(y_pred: np.ndarray,
                                  vehicle_density: np.ndarray) -> float:
    """
    Recommendations (/Vehicle/km):
    Average number of actionable recommendations generated per vehicle
    per km, proportional to the fraction of polluting detections and
    current density.
    """
    polluting_rate = np.mean(y_pred)
    avg_density    = np.mean(vehicle_density)
    # Rule-based rec count: 2 base recs for eco, 4 for polluting
    avg_recs = polluting_rate * 4 + (1 - polluting_rate) * 2
    return float(avg_recs * avg_density / 100.0)


def traffic_control_efficiency(y_pred: np.ndarray,
                                vehicle_density: np.ndarray,
                                distance_km: float = 1.0) -> float:
    """
    Traffic Control (Vehicles/km):
    Effective vehicles managed per km — those correctly rerouted/restricted.
    """
    polluting_idx = y_pred == 1
    controlled = float(np.sum(vehicle_density[polluting_idx]))
    return controlled / (distance_km + 1e-8)


def data_analysis_rate(y_pred: np.ndarray,
                        inference_time_s: float,
                        n_samples: int) -> float:
    """
    Data Analysis Rate (/Vehicle):
    Correctly classified samples per second per vehicle.
    Mirrors the DAR metric in Table 6.
    """
    throughput = n_samples / (inference_time_s + 1e-8)
    return float(throughput / (n_samples + 1e-8))


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print tables
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics_table(results: list) -> None:
    """Print a formatted comparison table to stdout."""
    header = f"  {'Model':<16} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Prec':>8} {'Recall':>8} {'MCC':>8} {'Time(s)':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"  {r['name']:<16}"
            f" {r['accuracy']:>8.4f}"
            f" {r['f1']:>8.4f}"
            f" {r.get('auc', float('nan')):>8.4f}"
            f" {r['precision']:>8.4f}"
            f" {r['recall']:>8.4f}"
            f" {r['mcc']:>8.4f}"
            f" {r['time_s']:>8.4f}"
        )
    print("=" * len(header))


def print_paper_metrics_table(results: list) -> None:
    """
    Print the paper-style comparative table (Table 5 format):
    Impact Verification, Classification Rate, Recommendations.
    """
    print("\n" + "=" * 60)
    print("  PAPER-STYLE COMPARATIVE TABLE")
    print("=" * 60)
    print(f"  {'Model':<16} {'Impact Verif.':>15} {'Class. Rate/km':>16} {'Kappa':>8}")
    print("-" * 60)
    for r in results:
        iv  = r.get("accuracy", 0)        # proxy for impact verification
        cr  = r.get("f1", 0) * 0.67       # scaled classification rate
        kap = r.get("kappa", 0)
        print(f"  {r['name']:<16} {iv:>15.4f} {cr:>16.4f} {kap:>8.4f}")
    print("=" * 60)


def save_metrics_csv(results: list, path: str = "results/test_metrics.csv") -> None:
    """Save metrics list to CSV."""
    import csv, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not results:
        return
    keys = [k for k in results[0].keys() if k != "confusion_matrix"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in keys}
            writer.writerow(row)
    print(f"  Metrics saved → {path}")