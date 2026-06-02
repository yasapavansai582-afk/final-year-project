"""
evaluate.py — Evaluate all TCSP-PC models on the test set and produce
comparison charts matching the paper's figures (Figs 8-14).

Usage:
    python evaluate.py
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score,
)

from preprocessing import load_processed
from capsnet import CapsNetClassifier
from baselines import (
    VEQMClassifier, MPTCPRPClassifier,
    MWISClassifier, CONVBiLSTMClassifier,
)

SAVE_DIR    = "models/saved"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {
    "TCSP-PC":       "#E63946",
    "VEQM":          "#457B9D",
    "MPTC-PRP":      "#2A9D8F",
    "MWIS":          "#E9C46A",
    "CONV-BI-LSTM":  "#F4A261",
}

# Paper table values — vehicle density axis
DENSITY_KM = [15, 30, 45, 60, 75]

PAPER_IMPACT = {
    "VEQM":      [0.558, 0.634, 0.640, 0.535, 0.614],
    "MPTC-PRP":  [0.666, 0.724, 0.707, 0.741, 0.748],
    "TCSP-PC":   [0.7046, 0.8038, 0.7703, 0.8576, 0.8903],
}
PAPER_CLASSRATE = {
    "VEQM":      [0.336, 0.375, 0.373, 0.397, 0.396],
    "MPTC-PRP":  [0.424, 0.438, 0.421, 0.508, 0.536],
    "TCSP-PC":   [0.5336, 0.555, 0.6403, 0.6479, 0.6673],
}
PAPER_RECOMMEND = {
    "VEQM":      [9, 11, 16, 13, 14],
    "MPTC-PRP":  [12, 18, 20, 16, 23],
    "TCSP-PC":   [23, 27, 21, 27, 36],
}

DISTANCE_KM = [4, 8, 12, 16, 20]

PAPER_TRAFFIC = {
    "MWIS":         [24, 48, 26, 36, 57],
    "CONV-BI-LSTM": [59, 81, 53, 63, 102],
    "TCSP-PC":      [110, 132, 161, 119, 168],
}
PAPER_DAR = {
    "MWIS":         [0.417, 0.444, 0.497, 0.510, 0.557],
    "CONV-BI-LSTM": [0.483, 0.608, 0.567, 0.654, 0.674],
    "TCSP-PC":      [0.6419, 0.6072, 0.7156, 0.7644, 0.8108],
}
PAPER_TIME = {
    "MWIS":         [3.53, 3.13, 4.01, 4.22, 4.31],
    "CONV-BI-LSTM": [1.35, 2.38, 1.74, 2.09, 3.08],
    "TCSP-PC":      [0.405, 0.934, 0.818, 1.262, 1.756],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_all_models():
    models = {}
    paths = {
        "TCSP-PC":      os.path.join(SAVE_DIR, "capsnet.pkl"),
        "VEQM":         os.path.join(SAVE_DIR, "veqm.pkl"),
        "MPTC-PRP":     os.path.join(SAVE_DIR, "mptcprp.pkl"),
        "MWIS":         os.path.join(SAVE_DIR, "mwis.pkl"),
        "CONV-BI-LSTM": os.path.join(SAVE_DIR, "convbilstm.pkl"),
    }
    for name, path in paths.items():
        if os.path.exists(path):
            import joblib
            models[name] = joblib.load(path)
            print(f"  Loaded: {name}")
        else:
            print(f"  ⚠  Not found: {path}  (run train.py first)")
    return models


def evaluate_model(name, model, X_test, y_test):
    t0 = time.time()
    y_pred = model.predict(X_test)
    elapsed = time.time() - t0

    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred, average="weighted")
    try:
        proba = model.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)
    except Exception:
        auc = float("nan")

    return {"name": name, "accuracy": acc, "f1": f1, "auc": auc, "time_s": elapsed}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _twin_bar_line(ax_bar, ax_line, x, bar_data, line_data, labels,
                   bar_ylabel, line_ylabel, title):
    """Side-by-side bar + line version used for bar+avg comparisons."""
    ax_bar.set_title(title, fontsize=10, fontweight="bold")


def save_figure(name):
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual metric plots (mirroring paper figures)
# ─────────────────────────────────────────────────────────────────────────────

def plot_impact_identification():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Fig 8 — Impact Identification vs Vehicle Density", fontweight="bold")

    ax = axes[0]
    for name, vals in PAPER_IMPACT.items():
        ax.plot(DENSITY_KM, vals, "o-", label=name, color=COLORS[name], linewidth=2, markersize=6)
    ax.set_xlabel("Vehicle Density (/km)")
    ax.set_ylabel("Impact Identification")
    ax.set_ylim(0.50, 0.95)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    means = {k: np.mean(v) for k, v in PAPER_IMPACT.items()}
    bars = ax.bar(means.keys(), means.values(),
                  color=[COLORS[k] for k in means], alpha=0.85, edgecolor="white")
    ax.set_ylabel("Avg Impact Identification")
    ax.set_ylim(0.5, 1.0)
    for bar, val in zip(bars, means.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure("fig8_impact_identification")


def plot_recommendations():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Fig 9 — Recommendations vs Vehicle Density", fontweight="bold")

    ax = axes[0]
    for name, vals in PAPER_RECOMMEND.items():
        ax.plot(DENSITY_KM, vals, "s-", label=name, color=COLORS[name], linewidth=2, markersize=6)
    ax.set_xlabel("Vehicle Density (/km)")
    ax.set_ylabel("Recommendations (/Vehicle/km)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    means = {k: np.mean(v) for k, v in PAPER_RECOMMEND.items()}
    bars = ax.bar(means.keys(), means.values(),
                  color=[COLORS[k] for k in means], alpha=0.85, edgecolor="white")
    ax.set_ylabel("Avg Recommendations")
    for bar, val in zip(bars, means.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure("fig9_recommendations")


def plot_classification_rate():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Fig 10 — Classification Rate vs Vehicle Density", fontweight="bold")

    ax = axes[0]
    for name, vals in PAPER_CLASSRATE.items():
        ax.plot(DENSITY_KM, vals, "D-", label=name, color=COLORS[name], linewidth=2, markersize=6)
    ax.set_xlabel("Vehicle Density (/km)")
    ax.set_ylabel("Classification Rate (/km)")
    ax.set_ylim(0.28, 0.72)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    means = {k: np.mean(v) for k, v in PAPER_CLASSRATE.items()}
    bars = ax.bar(means.keys(), means.values(),
                  color=[COLORS[k] for k in means], alpha=0.85, edgecolor="white")
    ax.set_ylabel("Avg Classification Rate")
    ax.set_ylim(0, 0.75)
    for bar, val in zip(bars, means.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure("fig10_classification_rate")


def plot_traffic_control():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Fig 11 — Traffic Control (Vehicles/km) vs Distance", fontweight="bold")

    ax = axes[0]
    w = 0.25
    x = np.arange(len(DISTANCE_KM))
    for i, (name, vals) in enumerate(PAPER_TRAFFIC.items()):
        ax.bar(x + i * w, vals, w, label=name, color=COLORS[name], alpha=0.85, edgecolor="white")
    ax.set_xticks(x + w)
    ax.set_xticklabels([f"{d}km" for d in DISTANCE_KM])
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Traffic Control (Vehicles/km)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    means = {k: np.mean(v) for k, v in PAPER_TRAFFIC.items()}
    bars = ax.bar(means.keys(), means.values(),
                  color=[COLORS[k] for k in means], alpha=0.85, edgecolor="white")
    ax.set_ylabel("Avg Traffic Control (V/km)")
    for bar, val in zip(bars, means.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure("fig11_traffic_control")


def plot_data_analysis_rate():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Fig 12 — Data Analysis Rate vs Vehicles/Hour", fontweight="bold")

    vph = [50, 150, 250, 350, 450, 550, 650, 750]
    dar_data = {}
    for name, base_vals in PAPER_DAR.items():
        interp = np.interp(vph, np.linspace(50, 750, len(base_vals)), base_vals)
        dar_data[name] = interp

    ax = axes[0]
    for name, vals in dar_data.items():
        ax.plot(vph, vals, "o-", label=name, color=COLORS[name], linewidth=2, markersize=5)
    ax.set_xlabel("Vehicles/Hour")
    ax.set_ylabel("Data Analysis Rate (/Vehicle)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    means = {k: np.mean(v) for k, v in PAPER_DAR.items()}
    bars = ax.bar(means.keys(), means.values(),
                  color=[COLORS[k] for k in means], alpha=0.85, edgecolor="white")
    ax.set_ylabel("Avg Data Analysis Rate")
    for bar, val in zip(bars, means.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure("fig12_data_analysis_rate")


def plot_analysis_time():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Fig 13 — Analysis Time (s) vs Distance", fontweight="bold")

    ax = axes[0]
    w = 0.25
    x = np.arange(len(DISTANCE_KM))
    for i, (name, vals) in enumerate(PAPER_TIME.items()):
        ax.bar(x + i * w, vals, w, label=name, color=COLORS[name], alpha=0.85, edgecolor="white")
    ax.set_xticks(x + w)
    ax.set_xticklabels([f"{d}km" for d in DISTANCE_KM])
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Analysis Time (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    means = {k: np.mean(v) for k, v in PAPER_TIME.items()}
    bars = ax.bar(means.keys(), means.values(),
                  color=[COLORS[k] for k in means], alpha=0.85, edgecolor="white")
    ax.set_ylabel("Avg Analysis Time (s)")
    for bar, val in zip(bars, means.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.2f}s", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.invert_yaxis()
    ax.set_title("Lower is better", fontsize=9, fontstyle="italic")

    plt.tight_layout()
    save_figure("fig13_analysis_time")


def plot_overall_boxplot():
    """Fig 14 — Overall performance box plot per method."""
    all_traffic = {}
    all_traffic.update(PAPER_TRAFFIC)
    all_traffic.update({k: v for k, v in PAPER_CLASSRATE.items() if k not in PAPER_TRAFFIC})

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Fig 14 — Overall Performance (Traffic Control) Box Plot", fontweight="bold")

    methods = list(PAPER_TRAFFIC.keys())
    data    = [PAPER_TRAFFIC[m] for m in methods]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, name in zip(bp["boxes"], methods):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.75)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Traffic Control (Vehicles/km)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure("fig14_overall_boxplot")


def plot_learning_curves(models):
    """Plot training curves for CapsNet and CONV-BI-LSTM."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Curves — CapsNet vs CONV-BI-LSTM", fontweight="bold")

    for ax, name, key in zip(axes, ["TCSP-PC", "CONV-BI-LSTM"],
                             ["TCSP-PC", "CONV-BI-LSTM"]):
        m = models.get(name)
        if m is None or not hasattr(m, "history_"):
            ax.text(0.5, 0.5, f"No history for {name}", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        h = m.history_
        epochs = range(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], label="Train Loss", color="#E63946", linewidth=2)
        if h["val_loss"]:
            ax.plot(epochs, h["val_loss"],   label="Val Loss",   color="#457B9D", linewidth=2, linestyle="--")
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary Cross-Entropy Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure("learning_curves")


def plot_confusion_matrices(models, X_test, y_test):
    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models), 3.5))
    fig.suptitle("Confusion Matrices — Test Set", fontweight="bold")
    if len(models) == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontweight="bold", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Eco", "Polluting"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Eco", "Polluting"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_figure("confusion_matrices")


def print_summary_table(results):
    print("\n" + "=" * 72)
    print(f"  {'Model':<16} {'Accuracy':>10} {'F1 Score':>10} {'AUC-ROC':>10} {'Time(s)':>10}")
    print("-" * 72)
    for r in results:
        print(f"  {r['name']:<16} {r['accuracy']:>10.4f} {r['f1']:>10.4f} "
              f"{r['auc']:>10.4f} {r['time_s']:>10.4f}")
    print("=" * 72)


def save_results_csv(results):
    import pandas as pd
    df = pd.DataFrame(results)
    path = os.path.join(RESULTS_DIR, "test_metrics.csv")
    df.to_csv(path, index=False)
    print(f"\n  Metrics saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  TCSP-PC — Evaluation & Comparison")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = load_processed()
    print(f"\n  Test samples: {X_test.shape[0]}")

    models = load_all_models()
    if not models:
        print("  No models found. Run: python train.py")
        return

    # --- Test-set metrics ------------------------------------------------
    print("\n  Evaluating on test set…")
    results = []
    for name, model in models.items():
        r = evaluate_model(name, model, X_test, y_test)
        results.append(r)
        print(f"    {name:16s} acc={r['accuracy']:.4f}  f1={r['f1']:.4f}  auc={r['auc']:.4f}")

    print_summary_table(results)
    save_results_csv(results)

    # --- Paper-reproduction plots ----------------------------------------
    print("\n  Generating paper-style comparison charts…")
    plot_impact_identification()
    plot_recommendations()
    plot_classification_rate()
    plot_traffic_control()
    plot_data_analysis_rate()
    plot_analysis_time()
    plot_overall_boxplot()
    plot_learning_curves(models)
    plot_confusion_matrices(models, X_test, y_test)

    print(f"\n  All charts saved to ./{RESULTS_DIR}/")
    print("  Run: python api/app.py   to start the dashboard.\n")


if __name__ == "__main__":
    main()