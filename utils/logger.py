import os
from dotenv import load_dotenv
import wandb
from typing import Dict, Any, Optional, List
from utils.plots import TrainingPlotter

load_dotenv()


class WandbLogger:
    """Weights & Biases logger for tracking experiments and visualizations."""

    def __init__(
        self, config, run_name: Optional[str] = None, notes: Optional[str] = None
    ):
        """Initialize W&B logger with config.

        Args:
            config: Config object with W&B settings
            run_name: Optional custom run name
            notes: Optional run notes/description
        """
        self.config = config
        self.plotter = TrainingPlotter()
        self.run = None

        api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=api_key)

        self.run = wandb.init(
            project=self.config["wandb"]["project"],
            entity=self.config["wandb"]["entity"],
            tags=self.config["wandb"]["tags"] if self.config["wandb"]["tags"] else [],
            name=self.config["wandb"]["run_name"],
            notes=notes,
            config={  # ADD THIS: Log config to W&B
                "model": self.config.get("model", {}),
                "training": self.config.get("training", {}),
                "lora": self.config.get("lora", {}),
                "data": self.config.get("data", {}),
            },
        )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log scalar metrics to W&B.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (epoch/iteration)
        """
        if self.run:
            wandb.log(metrics, step=step)

    def log_loss_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        epochs: Optional[List[int]] = None,
        step: Optional[int] = None,
    ):
        """Log training and validation loss curves.

        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            epochs: Optional list of epoch numbers
            step: Optional step number for logging
        """
        if self.run:
            fig = self.plotter.plot_loss_curves(train_losses, val_losses, epochs)
            wandb.log({"loss_curves": wandb.Image(fig)}, step=step)
            self.plotter.close_figure(fig)

    def log_metrics_plot(
        self,
        metrics_dict: Dict[str, List[float]],
        metric_name: str,
        step: Optional[int] = None,
    ):
        """Log metrics plot (e.g., accuracy, F1) over training.

        Args:
            metrics_dict: Dict with 'train' and 'val' keys containing metric lists
            metric_name: Name of the metric
            step: Optional step number for logging
        """
        if self.run:
            fig = self.plotter.plot_metrics(metrics_dict, metric_name)
            wandb.log({f"{metric_name.lower()}_plot": wandb.Image(fig)}, step=step)
            self.plotter.close_figure(fig)

    def log_confusion_matrix(
        self, cm, class_names: Optional[List[str]] = None, step: Optional[int] = None
    ):
        """Log confusion matrix visualization.

        Args:
            cm: Confusion matrix array
            class_names: Optional list of class names
            step: Optional step number for logging
        """
        if self.run:
            fig = self.plotter.plot_confusion_matrix(cm, class_names)
            wandb.log({"confusion_matrix": wandb.Image(fig)}, step=step)
            self.plotter.close_figure(fig)

    def log_learning_rate(
        self,
        learning_rates: List[float],
        steps: Optional[List[int]] = None,
        step: Optional[int] = None,
    ):
        """Log learning rate schedule.

        Args:
            learning_rates: List of learning rates
            steps: Optional list of step numbers
            step: Optional step number for logging
        """
        if self.run:
            fig = self.plotter.plot_learning_rate(learning_rates, steps)
            wandb.log({"learning_rate_schedule": wandb.Image(fig)}, step=step)
            self.plotter.close_figure(fig)

    def log_trainable_params(
        self, params_dict: Dict[str, float], step: Optional[int] = None
    ):
        """Log trainable parameters comparison.

        Args:
            params_dict: Dict with model names and parameter counts
            step: Optional step number for logging
        """
        if self.run:
            fig = self.plotter.plot_trainable_params(params_dict)
            wandb.log({"trainable_parameters": wandb.Image(fig)}, step=step)
            self.plotter.close_figure(fig)

    def log_model(self, model_path: str, name: str = "model"):
        """Log model artifact to W&B.

        Args:
            model_path: Path to saved model
            name: Artifact name
        """
        if self.run:
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

    def update_config(self, config_dict: Dict[str, Any]):
        """Update W&B run config with additional parameters.

        Args:
            config_dict: Dictionary of config parameters to add
        """
        if self.run:
            wandb.config.update(config_dict)

    def finish(self):
        """Finish the W&B run."""
        if self.run:
            wandb.finish()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
