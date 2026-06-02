"""
utils/visualizer.py — Plotting and chart utilities for TCSP-PC.

Wraps all matplotlib / seaborn figure generation behind clean functions
so that evaluate.py and notebooks can call them without boilerplate.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

# ── Defaults ─────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {
    "TCSP-PC":      "#E63946",
    "VEQM":         "#457B9D",
    "MPTC-PRP":     "#2A9D8F",
    "MWIS":         "#E9C46A",
    "CONV-BI-LSTM": "#F4A261",
}

_STYLE = {
    "figure.facecolor": "#1a1d27",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#2e3248",
    "axes.labelcolor":  "#e2e8f0",
    "axes.titlecolor":  "#e2e8f0",
    "xtick.color":      "#8892a4",
    "ytick.color":      "#8892a4",
    "grid.color":       "#2e3248",
    "text.color":       "#e2e8f0",
    "legend.facecolor": "#22263a",
    "legend.edgecolor": "#2e3248",
}


def _apply_style(ax):
    """Apply dark theme to a single Axes."""
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#8892a4")
    ax.spines[:].set_color("#2e3248")


def _save(name: str, dpi: int = 150) -> str:
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    plt.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor="#1a1d27", edgecolor="none")
    plt.close()
    print(f"  Saved → {path}")
    return path


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(model, model_name: str = "Model") -> str:
    """
    Plot train / val loss and accuracy for a model that exposes .history_.

    Parameters
    ----------
    model      : fitted model with .history_ dict
    model_name : display label
    """
    h = getattr(model, "history_", {})
    if not h or not h.get("train_loss"):
        print(f"  ⚠  No training history for {model_name}")
        return ""

    epochs = range(1, len(h["train_loss"]) + 1)
    has_val = bool(h.get("val_loss"))

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f"Training Curves — {model_name}", fontweight="bold",
                     color="#e2e8f0")

        # Loss
        ax = axes[0]
        ax.plot(epochs, h["train_loss"], label="Train Loss",
                color="#E63946", linewidth=2)
        if has_val:
            ax.plot(epochs, h["val_loss"], label="Val Loss",
                    color="#60a5fa", linewidth=2, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.set_title("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        _apply_style(ax)

        # Accuracy
        ax = axes[1]
        ax.plot(epochs, h["train_acc"], label="Train Acc",
                color="#4ade80", linewidth=2)
        if has_val and h.get("val_acc"):
            ax.plot(epochs, h["val_acc"], label="Val Acc",
                    color="#fbbf24", linewidth=2, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        _apply_style(ax)

        plt.tight_layout()
    return _save(f"training_curves_{model_name.lower().replace(' ', '_').replace('-', '')}")


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str = "Model",
                           class_names: list = None) -> str:
    """
    Plot a single confusion matrix with percentage annotations.
    """
    if class_names is None:
        class_names = ["Eco-friendly", "Polluting"]

    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.suptitle(f"Confusion Matrix — {model_name}", fontweight="bold",
                     color="#e2e8f0")

        im = ax.imshow(cm, cmap="Blues", vmin=0)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i,
                        f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                        ha="center", va="center", fontsize=11,
                        color="white" if cm[i, j] > thresh else "black",
                        fontweight="bold")
        _apply_style(ax)
        plt.tight_layout()
    return _save(f"cm_{model_name.lower().replace(' ', '_').replace('-', '')}")


def plot_all_confusion_matrices(models: dict,
                                 X_test: np.ndarray,
                                 y_test: np.ndarray) -> str:
    """Plot side-by-side confusion matrices for all models."""
    n = len(models)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4))
        fig.suptitle("Confusion Matrices — All Models", fontweight="bold",
                     color="#e2e8f0")
        if n == 1:
            axes = [axes]

        for ax, (name, model) in zip(axes, models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title(name, fontweight="bold", fontsize=9, color="#e2e8f0")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Eco", "Polluting"], fontsize=8)
            ax.set_yticklabels(["Eco", "Polluting"], fontsize=8)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            thresh = cm.max() / 2
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]),
                            ha="center", va="center", fontsize=12,
                            color="white" if cm[i, j] > thresh else "black",
                            fontweight="bold")
            _apply_style(ax)

        plt.tight_layout()
    return _save("confusion_matrices_all")


# ── ROC curves ────────────────────────────────────────────────────────────────

def plot_roc_curves(models: dict, X_test: np.ndarray,
                    y_test: np.ndarray) -> str:
    """
    Plot ROC curves for all models on the same axes.
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.suptitle("ROC Curves — Model Comparison", fontweight="bold",
                     color="#e2e8f0")

        for name, model in models.items():
            try:
                proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, proba)
                roc_auc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr,
                        color=COLORS.get(name, "#aaa"),
                        linewidth=2,
                        label=f"{name} (AUC={roc_auc_val:.3f})")
            except Exception:
                pass

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5,
                label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.05])
        _apply_style(ax)
        plt.tight_layout()
    return _save("roc_curves")


# ── Precision-Recall curves ───────────────────────────────────────────────────

def plot_precision_recall_curves(models: dict, X_test: np.ndarray,
                                  y_test: np.ndarray) -> str:
    """
    Plot Precision-Recall curves for all models.
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.suptitle("Precision-Recall Curves", fontweight="bold",
                     color="#e2e8f0")

        for name, model in models.items():
            try:
                proba = model.predict_proba(X_test)[:, 1]
                prec, rec, _ = precision_recall_curve(y_test, proba)
                ap = average_precision_score(y_test, proba)
                ax.plot(rec, prec,
                        color=COLORS.get(name, "#aaa"),
                        linewidth=2,
                        label=f"{name} (AP={ap:.3f})")
            except Exception:
                pass

        baseline = y_test.mean()
        ax.axhline(baseline, color="#8892a4", linestyle="--", linewidth=1,
                   label=f"Baseline ({baseline:.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        _apply_style(ax)
        plt.tight_layout()
    return _save("precision_recall_curves")


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list,
                              model_name: str = "MWIS / MPTC-PRP") -> str:
    """
    Plot feature importances for tree-based models (Random Forest, GBDT).
    Looks inside .rf_ or .gb_ attributes.
    """
    inner = getattr(model, "rf_", None) or getattr(model, "gb_", None)
    if inner is None or not hasattr(inner, "feature_importances_"):
        print(f"  ⚠  {model_name} has no feature_importances_")
        return ""

    imps = inner.feature_importances_
    n_extra = len(imps) - len(feature_names)
    names = list(feature_names) + [f"extra_{i}" for i in range(n_extra)]
    names = names[:len(imps)]

    idx = np.argsort(imps)[::-1]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle(f"Feature Importance — {model_name}", fontweight="bold",
                     color="#e2e8f0")
        bars = ax.bar(range(len(imps)),
                      imps[idx],
                      color=COLORS.get(model_name, "#60a5fa"),
                      alpha=0.85, edgecolor="#1a1d27")
        ax.set_xticks(range(len(imps)))
        ax.set_xticklabels([names[i] for i in idx], rotation=30,
                           ha="right", fontsize=8)
        ax.set_ylabel("Importance")
        ax.grid(True, alpha=0.3, axis="y")
        _apply_style(ax)
        plt.tight_layout()
    return _save(f"feature_importance_{model_name.lower().replace(' ', '_').replace('-', '')}")


# ── Emission distribution ─────────────────────────────────────────────────────

def plot_emission_distribution(df) -> str:
    """
    Plot emission_rate distribution split by class (eco vs polluting).
    df : pandas DataFrame with 'emission_rate' and 'emission_label' columns.
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle("Emission Rate Distribution by Class", fontweight="bold",
                     color="#e2e8f0")

        eco  = df[df["emission_label"] == 0]["emission_rate"]
        poll = df[df["emission_label"] == 1]["emission_rate"]

        ax.hist(eco,  bins=50, alpha=0.65, color="#4ade80",
                label=f"Eco-friendly (n={len(eco):,})", density=True)
        ax.hist(poll, bins=50, alpha=0.65, color="#f87171",
                label=f"Polluting (n={len(poll):,})", density=True)
        ax.axvline(30, color="#fbbf24", linestyle="--", linewidth=1.5,
                   label="Threshold (30 g/km)")
        ax.set_xlabel("Emission Rate (g/km)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        _apply_style(ax)
        plt.tight_layout()
    return _save("emission_distribution")


# ── Bar comparison ────────────────────────────────────────────────────────────

def plot_metric_comparison(results: list, metric: str = "accuracy",
                            title: str = "") -> str:
    """
    Horizontal bar chart comparing a single metric across all models.

    Parameters
    ----------
    results : list of dicts from compute_all_metrics()
    metric  : key in the results dicts
    title   : optional chart title
    """
    names = [r["name"] for r in results]
    vals  = [float(r.get(metric, 0)) for r in results]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.7)))
        title = title or f"Model Comparison — {metric.replace('_', ' ').title()}"
        fig.suptitle(title, fontweight="bold", color="#e2e8f0")

        colors = [COLORS.get(n, "#60a5fa") for n in names]
        bars = ax.barh(names, vals, color=colors, alpha=0.85, edgecolor="#1a1d27",
                       height=0.55)
        for bar, val in zip(bars, vals):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, color="#e2e8f0")

        ax.set_xlim(0, max(vals) * 1.15)
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3, axis="x")
        _apply_style(ax)
        plt.tight_layout()
    return _save(f"comparison_{metric}")


# ── Heatmap ───────────────────────────────────────────────────────────────────

def plot_metrics_heatmap(results: list) -> str:
    """
    Seaborn heatmap of all numeric metrics across models.
    """
    import pandas as pd
    skip = {"name", "confusion_matrix", "time_s"}
    cols = [k for k in results[0].keys()
            if k not in skip and not isinstance(results[0][k], list)]
    df = pd.DataFrame([{c: float(r[c]) for c in cols} for r in results],
                      index=[r["name"] for r in results])

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.2), max(4, len(results))))
        fig.suptitle("Metrics Heatmap — All Models", fontweight="bold",
                     color="#e2e8f0")

        sns.heatmap(df, ax=ax, annot=True, fmt=".3f", cmap="YlGn",
                    linewidths=0.5, linecolor="#2e3248",
                    cbar_kws={"shrink": 0.8})
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
    return _save("metrics_heatmap")