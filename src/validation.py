from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras

logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_proba: Predicted probabilities

    Returns:
        Dictionary with all metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        raise ValueError(f"Expected binary confusion matrix, got shape {cm.shape}")
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    logger.info("Metrics computed: accuracy=%.4f, f1=%.4f, auc_roc=%.4f", 
                metrics["accuracy"], metrics["f1_score"], metrics["auc_roc"])
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1"
) -> tuple[float, float]:
    """
    Find optimal classification threshold by maximizing a metric.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        (optimal_threshold, optimal_metric_value)
    """
    thresholds = np.arange(0.1, 0.95, 0.01)
    scores = []

    for threshold in thresholds:
        y_pred_temp = (y_proba >= threshold).astype(int)
        
        if metric == "f1":
            score = f1_score(y_true, y_pred_temp, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred_temp, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred_temp, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)

    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]

    logger.info(
        "Optimal threshold found: %.3f (%.4f %s)", optimal_threshold, optimal_score, metric
    )
    return optimal_threshold, optimal_score


def plot_training_history(history: keras.callbacks.History, output_dir: str | Path) -> None:
    """
    Plot training history (loss and metrics).

    Args:
        history: Keras History object from model.fit()
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    # Loss
    axes[0, 0].plot(epochs, history_dict.get("loss", []), "b-", label="Training Loss", linewidth=2)
    axes[0, 0].plot(
        epochs, history_dict.get("val_loss", []), "r-", label="Validation Loss", linewidth=2
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history_dict.get("accuracy", []), "b-", label="Training", linewidth=2)
    axes[0, 1].plot(
        epochs, history_dict.get("val_accuracy", []), "r-", label="Validation", linewidth=2
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision
    axes[0, 2].plot(epochs, history_dict.get("precision", []), "b-", label="Training", linewidth=2)
    axes[0, 2].plot(
        epochs, history_dict.get("val_precision", []), "r-", label="Validation", linewidth=2
    )
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Precision")
    axes[0, 2].set_title("Precision")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Recall
    axes[1, 0].plot(epochs, history_dict.get("recall", []), "b-", label="Training", linewidth=2)
    axes[1, 0].plot(
        epochs, history_dict.get("val_recall", []), "r-", label="Validation", linewidth=2
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].set_title("Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # AUC-ROC
    axes[1, 1].plot(epochs, history_dict.get("auc_roc", []), "b-", label="Training", linewidth=2)
    axes[1, 1].plot(
        epochs, history_dict.get("val_auc_roc", []), "r-", label="Validation", linewidth=2
    )
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("AUC-ROC")
    axes[1, 1].set_title("AUC-ROC")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # AUC-PR
    axes[1, 2].plot(epochs, history_dict.get("auc_pr", []), "b-", label="Training", linewidth=2)
    axes[1, 2].plot(
        epochs, history_dict.get("val_auc_pr", []), "r-", label="Validation", linewidth=2
    )
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("AUC-PR")
    axes[1, 2].set_title("AUC-PR (Precision-Recall)")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "training_history.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Training history plot saved to %s", plot_path)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, output_dir: str | Path
) -> None:
    """
    Plot confusion matrix with annotations.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Directory to save plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    # Add metrics text
    text = f"TN: {tn:,}\nFP: {fp:,}\nFN: {fn:,}\nTP: {tp:,}"
    plt.text(
        0.02, 0.98, text, transform=plt.gca().transAxes,
        fontsize=11, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    plt.tight_layout()
    plot_path = output_dir / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Confusion matrix plot saved to %s", plot_path)


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, output_dir: str | Path) -> None:
    """
    Plot ROC curve with AUC score.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        output_dir: Directory to save plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "roc_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("ROC curve plot saved to %s", plot_path)


def plot_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, output_dir: str | Path) -> None:
    """
    Plot Precision-Recall curve with Average Precision score.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        output_dir: Directory to save plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()

    plt.figure(figsize=(10, 8))
    plt.plot(recall_vals, precision_vals, color="green", lw=2, label=f"PR curve (AP = {pr_auc:.4f})")
    plt.axhline(y=baseline, color="navy", lw=2, linestyle="--", 
                label=f"Baseline (Prevalence = {baseline:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "precision_recall_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Precision-Recall curve plot saved to %s", plot_path)


def plot_threshold_analysis(
    y_true: np.ndarray, y_proba: np.ndarray, output_dir: str | Path
) -> None:
    """
    Plot F1, Precision, and Recall vs threshold.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        output_dir: Directory to save plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = np.arange(0.1, 0.95, 0.01)
    f1_scores = []
    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred_temp = (y_proba >= threshold).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_temp, zero_division=0))
        precisions.append(precision_score(y_true, y_pred_temp, zero_division=0))
        recalls.append(recall_score(y_true, y_pred_temp, zero_division=0))

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, f1_scores, "b-", linewidth=2, label="F1-Score")
    plt.plot(thresholds, precisions, "g-", linewidth=2, label="Precision")
    plt.plot(thresholds, recalls, "r-", linewidth=2, label="Recall")
    plt.axvline(x=optimal_threshold, color="orange", linestyle="--", linewidth=2,
                label=f"Optimal Threshold = {optimal_threshold:.3f}")
    plt.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Default Threshold = 0.50")
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Metrics vs Classification Threshold", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "threshold_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Threshold analysis plot saved to %s", plot_path)


def generate_classification_report_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate text classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Classification report as string
    """
    report = classification_report(
        y_true, y_pred, target_names=["Legitimate", "Fraud"], digits=4
    )
    return report


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str | Path,
    threshold: float = 0.5,
    figures_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Complete model evaluation with metrics and plots.

    This is the main evaluation entry point called from main.py.

    Args:
        model: Trained Keras model
        X_test: Test features (already preprocessed)
        y_test: Test labels
        output_dir: Directory to save results
        threshold: Classification threshold

    Returns:
        Dictionary with all evaluation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = Path(figures_dir) if figures_dir is not None else output_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting model evaluation on %d test samples", len(X_test))

    # Get predictions
    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_proba >= threshold).astype(int)

    # Compute metrics
    metrics = compute_classification_metrics(y_test, y_pred, y_proba)

    # Find optimal threshold
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_proba, metric="f1")
    metrics["optimal_threshold"] = float(optimal_threshold)
    metrics["optimal_f1_score"] = float(optimal_f1)

    # Generate classification report
    report_text = generate_classification_report_text(y_test, y_pred)
    metrics["classification_report"] = report_text

    # Save metrics to JSON
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        # Convert report to save properly
        metrics_to_save = metrics.copy()
        json.dump(metrics_to_save, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Generate all plots
    plot_confusion_matrix(y_test, y_pred, figures_dir)
    plot_roc_curve(y_test, y_proba, figures_dir)
    plot_pr_curve(y_test, y_proba, figures_dir)
    plot_threshold_analysis(y_test, y_proba, figures_dir)

    # Save classification report as text
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(report_text)
        f.write("\n" + "="*70 + "\n")
    logger.info("Classification report saved to %s", report_path)

    logger.info("Evaluation completed successfully")
    return metrics
