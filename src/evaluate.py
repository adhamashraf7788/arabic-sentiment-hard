import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


def compute_metrics(y_true, y_pred, model_name: str = "Model") -> dict:
    """Compute all metrics and return as dict."""
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_negative": f1_score(y_true, y_pred, average=None)[0],
        "f1_positive": f1_score(y_true, y_pred, average=None)[1],
    }
    return metrics


def print_report(y_true, y_pred, model_name: str = "Model"):
    """Print full classification report."""
    print(f"\n{'='*50}")
    print(f"  {model_name} — Classification Report")
    print(f"{'='*50}")
    print(classification_report(
        y_true, y_pred,
        target_names=["Negative", "Positive"]
    ))


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model", save_path: str = None):
    """Plot and optionally save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_model_comparison(results: list[dict], save_path: str = None):
    """
    Bar chart comparing F1 scores across models.
    results: list of dicts from compute_metrics()
    """
    df = pd.DataFrame(results)
    df = df.set_index("model")

    metrics_to_plot = ["f1_weighted", "f1_negative", "f1_positive"]
    labels = ["F1 Weighted", "F1 Negative Class", "F1 Positive Class"]

    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, label) in enumerate(zip(metrics_to_plot, labels)):
        ax.bar(x + i * width, df[metric], width, label=label)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df.index, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Experiment C (Trained Balanced, Tested Unbalanced)")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def error_analysis(df_test: pd.DataFrame, y_true, y_pred, n: int = 10) -> pd.DataFrame:
    """
    Return n misclassified examples for manual inspection.
    df_test must have a 'review' column.
    """
    df = df_test.copy().reset_index(drop=True)
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["correct"] = df["y_true"] == df["y_pred"]

    misclassified = df[~df["correct"]].sample(
        n=min(n, (~df["correct"]).sum()),
        random_state=42
    )[["review", "y_true", "y_pred"]]

    label_map = {0: "Negative", 1: "Positive"}
    misclassified["y_true"] = misclassified["y_true"].map(label_map)
    misclassified["y_pred"] = misclassified["y_pred"].map(label_map)

    return misclassified