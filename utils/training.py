"""
YOLO Training Utilities
=======================
Trainer class that wraps YOLO training with custom logging and callbacks.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import yaml

from ultralytics import YOLO
from utils.logger import WandbLogger
from utils.data_loader import DatasetManager


class YOLOTrainer:
    """
    Trainer class for YOLO object detection models.
    
    Wraps ultralytics training with:
    - Wandb logging integration
    - Configurable training parameters
    - GPU metrics tracking
    - Checkpoint management
    """

    def __init__(self, model: YOLO, config: Dict[str, Any], data_yaml_path: str):
        """
        Initialize trainer.

        Args:
            model: YOLO model instance
            config: Configuration dictionary
            data_yaml_path: Path to data.yaml file
        """
        self.model = model
        self.config = config
        self.data_yaml_path = data_yaml_path
        
        # Extract configs
        self.training_config = config.get("training", {})
        self.augmentation_config = config.get("augmentation", {})
        self.output_config = config.get("output", {})
        self.wandb_config = config.get("wandb", {})
        
        # Initialize logger if wandb is enabled
        self.logger = None
        if self.wandb_config.get("enabled", True):
            self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        run_name = self.wandb_config.get("run_name")
        if run_name is None:
            # Auto-generate run name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.config.get("model", {}).get("name", "yolo")
            run_name = f"{model_name}_{timestamp}"
            
        self.logger = WandbLogger(self.config, run_name=run_name)

    def _build_train_args(self) -> Dict[str, Any]:
        """
        Build training arguments from config.

        Returns:
            Dictionary of training arguments for YOLO
        """
        # Basic training parameters
        train_args = {
            "data": self.data_yaml_path,
            "epochs": self.training_config.get("epochs", 100),
            "imgsz": self.training_config.get("imgsz", 640),
            "batch": self.training_config.get("batch_size", 16),
            "patience": self.training_config.get("patience", 50),
            "save_period": self.training_config.get("save_period", 10),
            "device": self.training_config.get("device"),
            "workers": self.training_config.get("workers", 8),
            "pretrained": self.config.get("model", {}).get("pretrained", True),
            "seed": self.config.get("seed", 42),
        }
        
        # Optimizer settings
        optimizer_config = self.training_config.get("optimizer", {})
        train_args.update({
            "optimizer": optimizer_config.get("name", "auto"),
            "lr0": optimizer_config.get("lr0", 0.01),
            "lrf": optimizer_config.get("lrf", 0.01),
            "momentum": optimizer_config.get("momentum", 0.937),
            "weight_decay": optimizer_config.get("weight_decay", 0.0005),
            "warmup_epochs": optimizer_config.get("warmup_epochs", 3.0),
            "warmup_momentum": optimizer_config.get("warmup_momentum", 0.8),
            "warmup_bias_lr": optimizer_config.get("warmup_bias_lr", 0.1),
        })
        
        # Augmentation settings
        if self.augmentation_config.get("enabled", True):
            train_args.update({
                "augment": True,
                "hsv_h": self.augmentation_config.get("hsv_h", 0.015),
                "hsv_s": self.augmentation_config.get("hsv_s", 0.7),
                "hsv_v": self.augmentation_config.get("hsv_v", 0.4),
                "degrees": self.augmentation_config.get("degrees", 0.0),
                "translate": self.augmentation_config.get("translate", 0.1),
                "scale": self.augmentation_config.get("scale", 0.5),
                "shear": self.augmentation_config.get("shear", 0.0),
                "perspective": self.augmentation_config.get("perspective", 0.0),
                "flipud": self.augmentation_config.get("flipud", 0.0),
                "fliplr": self.augmentation_config.get("fliplr", 0.5),
                "mosaic": self.augmentation_config.get("mosaic", 1.0),
                "mixup": self.augmentation_config.get("mixup", 0.0),
                "copy_paste": self.augmentation_config.get("copy_paste", 0.0),
            })
        else:
            train_args["augment"] = False
        
        # Output settings
        train_args.update({
            "project": self.output_config.get("project", "runs/detect"),
            "name": self.output_config.get("name", "train"),
            "exist_ok": self.output_config.get("exist_ok", True),
            "save": self.output_config.get("save", True),
            "save_txt": self.output_config.get("save_txt", True),
            "save_conf": self.output_config.get("save_conf", True),
            "plots": self.output_config.get("plots", True),
        })
        
        # Wandb integration (ultralytics has built-in wandb support)
        if self.wandb_config.get("enabled", True):
            os.environ["WANDB_PROJECT"] = self.wandb_config.get("project", "PCB_Defect_Detection")
            if self.wandb_config.get("entity"):
                os.environ["WANDB_ENTITY"] = self.wandb_config["entity"]
        
        return train_args

    def train(self) -> Any:
        """
        Run training.

        Returns:
            Training results from YOLO
        """
        print("\n" + "=" * 60)
        print("  Starting YOLO Training")
        print("=" * 60)
        
        # Build training arguments
        train_args = self._build_train_args()
        
        # Print configuration summary
        self._print_config_summary(train_args)
        
        # Run training
        results = self.model.train(**train_args)
        
        # Log final results
        if self.logger:
            self._log_final_results(results)
            self.logger.finish()
        
        print("\n" + "=" * 60)
        print("  Training Complete!")
        print("=" * 60)
        print(f"  Best weights: {results.save_dir}/weights/best.pt")
        print(f"  Results: {results.save_dir}")
        print("=" * 60 + "\n")
        
        return results

    def _print_config_summary(self, train_args: Dict[str, Any]) -> None:
        """Print training configuration summary."""
        print(f"\nConfiguration:")
        print(f"  Model: {self.config.get('model', {}).get('name', 'unknown')}")
        print(f"  Data: {self.data_yaml_path}")
        print(f"  Epochs: {train_args['epochs']}")
        print(f"  Batch size: {train_args['batch']}")
        print(f"  Image size: {train_args['imgsz']}")
        print(f"  Device: {train_args.get('device', 'auto')}")
        print(f"  Optimizer: {train_args['optimizer']}")
        print(f"  Initial LR: {train_args['lr0']}")
        print(f"  Augmentation: {'Enabled' if train_args.get('augment', True) else 'Disabled'}")
        
        freeze = self.config.get("model", {}).get("freeze_strategy", "none")
        print(f"  Freeze strategy: {freeze}")
        print("=" * 60 + "\n")

    def _log_final_results(self, results: Any) -> None:
        """Log final training results to wandb."""
        if not self.logger:
            return
            
        try:
            metrics = {
                "final/mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                "final/mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
                "final/precision": results.results_dict.get("metrics/precision(B)", 0),
                "final/recall": results.results_dict.get("metrics/recall(B)", 0),
            }
            self.logger.log_metrics(metrics)
        except Exception as e:
            print(f"Warning: Could not log final metrics: {e}")


class TrainerFactory:
    """Factory for creating trainers with different configurations."""

    @staticmethod
    def create_trainer(
        config_path: str,
        data_path: Optional[str] = None,
        checkpoint: Optional[str] = None,
    ) -> YOLOTrainer:
        """
        Create a trainer from config file.

        Args:
            config_path: Path to config YAML file
            data_path: Optional override for data path
            checkpoint: Optional checkpoint to resume from

        Returns:
            Configured YOLOTrainer instance
        """
        from model.model_loader import ModelLoader
        
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Override data path if provided
        if data_path:
            config["data"]["root"] = data_path
        
        # Create data.yaml
        data_manager = DatasetManager(config)
        data_yaml_path = data_manager.create_data_yaml()
        
        # Load model
        model_loader = ModelLoader(config)
        model = model_loader.load_model(checkpoint_path=checkpoint)
        
        # Create trainer
        trainer = YOLOTrainer(model, config, data_yaml_path)
        
        return trainer
