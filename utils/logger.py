"""
Weights & Biases Logger for YOLO Training
==========================================
Handles experiment tracking and visualization logging.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from dotenv import load_dotenv
import wandb

from utils.plots import TrainingPlotter

load_dotenv()


class WandbLogger:
    """Weights & Biases logger for tracking YOLO training experiments."""

    def __init__(
        self, 
        config: Dict[str, Any], 
        run_name: Optional[str] = None,
        notes: Optional[str] = None
    ):
        """
        Initialize W&B logger.

        Args:
            config: Configuration dictionary with wandb settings
            run_name: Optional custom run name
            notes: Optional run notes/description
        """
        self.config = config
        self.wandb_config = config.get("wandb", {})
        self.plotter = TrainingPlotter()
        self.run = None

        if not self.wandb_config.get("enabled", True):
            print("Wandb logging disabled")
            return

        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        else:
            print("Warning: WANDB_API_KEY not set. Run 'wandb login' to authenticate.")

        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = config.get("model", {}).get("name", "yolo")
            run_name = f"{model_name}_{timestamp}"

        try:
            self.run = wandb.init(
                project=self.wandb_config.get("project", "PCB_Defect_Detection"),
                entity=self.wandb_config.get("entity"),
                tags=self.wandb_config.get("tags", ["yolo", "pcb"]),
                name=run_name,
                notes=notes,
                config={
                    "model": config.get("model", {}),
                    "training": config.get("training", {}),
                    "augmentation": config.get("augmentation", {}),
                    "data": config.get("data", {}),
                },
                mode=self.wandb_config.get("mode", "online"),
            )
            print(f"Wandb run initialized: {run_name}")
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            self.run = None

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log scalar metrics to W&B."""
        if self.run:
            wandb.log(metrics, step=step)

    def log_loss_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        epochs: Optional[List[int]] = None,
        step: Optional[int] = None,
    ) -> None:
        """Log training and validation loss curves."""
        if self.run:
            fig = self.plotter.plot_loss_curves(train_losses, val_losses, epochs)
            wandb.log({"loss_curves": wandb.Image(fig)}, step=step)
            self.plotter.close_figure(fig)

    def log_detection_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log object detection specific metrics."""
        if self.run:
            prefixed = {f"detection/{k}": v for k, v in metrics.items()}
            wandb.log(prefixed, step=step)

    def log_confusion_matrix(
        self, cm, class_names: Optional[List[str]] = None, step: Optional[int] = None
    ) -> None:
        """Log confusion matrix visualization."""
        if self.run:
            fig = self.plotter.plot_confusion_matrix(cm, class_names)
            wandb.log({"confusion_matrix": wandb.Image(fig)}, step=step)
            self.plotter.close_figure(fig)

    def log_class_distribution(
        self,
        distribution: Dict[str, int],
        title: str = "Class Distribution",
        step: Optional[int] = None,
    ) -> None:
        """Log class distribution bar chart."""
        if self.run:
            fig = self.plotter.plot_class_distribution(distribution, title)
            wandb.log({"class_distribution": wandb.Image(fig)}, step=step)
            self.plotter.close_figure(fig)

    def log_model(self, model_path: str, name: str = "model") -> None:
        """Log model artifact to W&B."""
        if self.run:
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            print(f"Logged model artifact: {name}")

    def log_trainable_params(
        self, params_dict: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log trainable parameters comparison."""
        if self.run:
            fig = self.plotter.plot_trainable_params(params_dict)
            wandb.log({"trainable_parameters": wandb.Image(fig)}, step=step)
            self.plotter.close_figure(fig)

    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """Update W&B run config."""
        if self.run:
            wandb.config.update(config_dict)

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.run:
            wandb.finish()
            print("Wandb run finished")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
