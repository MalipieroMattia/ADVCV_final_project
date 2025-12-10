"""
Plotting Utilities for YOLO Training
=====================================
Create visualizations for training metrics and detection results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd


class TrainingPlotter:
    """Create plots for YOLO training metrics compatible with W&B logging."""

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """Initialize plotter with consistent styling."""
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("seaborn-darkgrid")
        sns.set_palette("husl")

    def plot_loss_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        epochs: Optional[List[int]] = None,
    ) -> plt.Figure:
        """Plot training and validation loss curves."""
        if epochs is None:
            epochs = range(1, len(train_losses) + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_losses, "o-", label="Train Loss", linewidth=2)
        ax.plot(epochs, val_losses, "s-", label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_metrics(
        self, metrics_dict: Dict[str, List[float]], metric_name: str = "Accuracy"
    ) -> plt.Figure:
        """Plot training and validation metrics over epochs."""
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(metrics_dict["train"]) + 1)

        ax.plot(
            epochs,
            metrics_dict["train"],
            "o-",
            label=f"Train {metric_name}",
            linewidth=2,
        )
        ax.plot(
            epochs, metrics_dict["val"], "s-", label=f"Val {metric_name}", linewidth=2
        )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f"{metric_name} Over Training", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_confusion_matrix(
        self, cm: np.ndarray, class_names: Optional[List[str]] = None
    ) -> plt.Figure:
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
        )
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        return fig

    def plot_class_distribution(
        self,
        distribution: Dict[str, int],
        title: str = "Class Distribution",
    ) -> plt.Figure:
        """Plot class distribution bar chart."""
        fig, ax = plt.subplots(figsize=(12, 6))

        classes = list(distribution.keys())
        counts = list(distribution.values())

        bars = ax.bar(classes, counts, color=sns.color_palette("husl", len(classes)))
        ax.set_xlabel("Defect Class", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def plot_learning_rate(
        self, learning_rates: List[float], steps: Optional[List[int]] = None
    ) -> plt.Figure:
        """Plot learning rate schedule."""
        if steps is None:
            steps = range(len(learning_rates))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, learning_rates, linewidth=2, color="#e74c3c")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        plt.tight_layout()
        return fig

    def plot_trainable_params(self, params_dict: Dict[str, float]) -> plt.Figure:
        """Visualize trainable parameters comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        models = list(params_dict.keys())
        params = list(params_dict.values())

        params_m = [p / 1e6 if p > 1e6 else p for p in params]

        colors = sns.color_palette("husl", len(models))
        bars = ax.barh(models, params_m, color=colors)
        ax.set_xlabel("Parameters (M)", fontsize=12)
        ax.set_title("Model Parameters", fontsize=14, fontweight="bold")

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}M",
                ha="left",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()
        return fig

    def plot_losses_from_results(self, results_csv_path: str) -> Optional[plt.Figure]:
        """Plot train/val losses from Ultralytics results.csv."""
        results_csv = Path(results_csv_path)
        if not results_csv.exists():
            return None

        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        if "epoch" in df.columns:
            epochs = df["epoch"]
        else:
            epochs = range(len(df))

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Training vs Validation Losses", fontsize=14, fontweight="bold")

        # Box loss
        ax = axes[0, 0]
        if "train/box_loss" in df.columns and "val/box_loss" in df.columns:
            ax.plot(epochs, df["train/box_loss"], label="Train Box", linewidth=2)
            ax.plot(epochs, df["val/box_loss"], label="Val Box", linewidth=2)
        ax.set_title("Box Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Class loss
        ax = axes[0, 1]
        if "train/cls_loss" in df.columns and "val/cls_loss" in df.columns:
            ax.plot(epochs, df["train/cls_loss"], label="Train Cls", linewidth=2)
            ax.plot(epochs, df["val/cls_loss"], label="Val Cls", linewidth=2)
        ax.set_title("Class Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # DFL loss
        ax = axes[1, 0]
        if "train/dfl_loss" in df.columns and "val/dfl_loss" in df.columns:
            ax.plot(epochs, df["train/dfl_loss"], label="Train DFL", linewidth=2)
            ax.plot(epochs, df["val/dfl_loss"], label="Val DFL", linewidth=2)
        ax.set_title("DFL Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Total loss (approx)
        ax = axes[1, 1]
        train_total = np.zeros(len(df))
        val_total = np.zeros(len(df))
        for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]:
            if col in df.columns:
                train_total += df[col].fillna(0)
        for col in ["val/box_loss", "val/cls_loss", "val/dfl_loss"]:
            if col in df.columns:
                val_total += df[col].fillna(0)
        ax.plot(epochs, train_total, label="Train Total", linewidth=2)
        ax.plot(epochs, val_total, label="Val Total", linewidth=2)
        ax.set_title("Total Loss (Approx)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_metrics_from_results(self, results_csv_path: str) -> Optional[plt.Figure]:
        """Plot precision/recall and mAP from Ultralytics results.csv."""
        results_csv = Path(results_csv_path)
        if not results_csv.exists():
            return None

        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        if "epoch" in df.columns:
            epochs = df["epoch"]
        else:
            epochs = range(len(df))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Validation Metrics", fontsize=14, fontweight="bold")

        # Precision & recall
        ax = axes[0]
        if "metrics/precision(B)" in df.columns:
            ax.plot(epochs, df["metrics/precision(B)"], label="Precision", linewidth=2)
        if "metrics/recall(B)" in df.columns:
            ax.plot(epochs, df["metrics/recall(B)"], label="Recall", linewidth=2)
        ax.set_title("Precision & Recall")
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend()

        # mAP
        ax = axes[1]
        if "metrics/mAP50(B)" in df.columns:
            ax.plot(epochs, df["metrics/mAP50(B)"], label="mAP@0.5", linewidth=2)
        if "metrics/mAP50-95(B)" in df.columns:
            ax.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", linewidth=2)
        ax.set_title("mAP")
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    @staticmethod
    def close_figure(fig: plt.Figure) -> None:
        """Close figure to free memory."""
        plt.close(fig)
