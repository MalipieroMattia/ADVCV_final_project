"""
Plotting Utilities for YOLO Training
=====================================
Create visualizations for training metrics and detection results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional


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

    @staticmethod
    def close_figure(fig: plt.Figure) -> None:
        """Close figure to free memory."""
        plt.close(fig)
