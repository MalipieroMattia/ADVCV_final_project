"""
YOLO Model Loader
=================
Handles loading and configuring YOLO models with different strategies:
- Different model sizes (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
- Pretrained vs from scratch
- Freeze backbone strategies
- Different YOLO versions (v8, v9, v10, v11)
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from ultralytics import YOLO
import torch


class ModelLoader:
    """Load and configure YOLO models based on configuration."""

    # Available YOLO model variants
    YOLO_VARIANTS = {
        # YOLOv8 - Standard
        "yolov8n": "yolov8n.pt",
        "yolov8s": "yolov8s.pt",
        "yolov8m": "yolov8m.pt",
        "yolov8l": "yolov8l.pt",
        "yolov8x": "yolov8x.pt",
        # YOLOv8 - P2 (Small Object Detection)
        # P2 adds an extra detection head at stride 4 (vs stride 8 for P3)
        # This preserves tiny features (<15px) that disappear in standard P5 architecture
        "yolov8n-p2": "yolov8n-p2.yaml",
        "yolov8s-p2": "yolov8s-p2.yaml",
        "yolov8m-p2": "yolov8m-p2.yaml",
        "yolov8l-p2": "yolov8l-p2.yaml",
        "yolov8x-p2": "yolov8x-p2.yaml",
        # YOLOv9
        "yolov9t": "yolov9t.pt",
        "yolov9s": "yolov9s.pt",
        "yolov9m": "yolov9m.pt",
        "yolov9c": "yolov9c.pt",
        "yolov9e": "yolov9e.pt",
        # YOLOv10
        "yolov10n": "yolov10n.pt",
        "yolov10s": "yolov10s.pt",
        "yolov10m": "yolov10m.pt",
        "yolov10l": "yolov10l.pt",
        "yolov10x": "yolov10x.pt",
        # YOLO11 (latest)
        "yolo11n": "yolo11n.pt",
        "yolo11s": "yolo11s.pt",
        "yolo11m": "yolo11m.pt",
        "yolo11l": "yolo11l.pt",
        "yolo11x": "yolo11x.pt",
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model loader with configuration.

        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.model_config = config.get("model", {})

    def load_model(self, checkpoint_path: Optional[str] = None) -> YOLO:
        """
        Load YOLO model based on configuration.

        Args:
            checkpoint_path: Optional path to resume from checkpoint

        Returns:
            YOLO model instance
        """
        # Resume from checkpoint if provided
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path}")
            return YOLO(checkpoint_path)

        # Get model variant from config
        model_name = self.model_config.get("name", "yolov8n")
        pretrained = self.model_config.get("pretrained", True)

        # Determine model path
        if pretrained:
            # Use pretrained weights
            if model_name in self.YOLO_VARIANTS:
                model_path = self.YOLO_VARIANTS[model_name]
            else:
                # Assume it's a direct path or standard name
                model_path = f"{model_name}.pt"
            print(f"Loading pretrained model: {model_path}")
        else:
            # Train from scratch using YAML config
            model_path = f"{model_name}.yaml"
            print(f"Training from scratch: {model_path}")

        # For P2 variants, start from architecture YAML and transfer matching
        # weights from the non-P2 pretrained checkpoint.
        if pretrained and model_name.endswith("-p2"):
            base_model_name = model_name.replace("-p2", "")
            base_weights = self.YOLO_VARIANTS.get(base_model_name, f"{base_model_name}.pt")
            print(f"Building {model_path} and transferring weights from {base_weights}")
            model = YOLO(model_path).load(base_weights)
        else:
            model = YOLO(model_path)

        # Print model info
        self.print_model_info(model)

        return model

    def _apply_freeze_strategy(self, model: YOLO, strategy: str) -> None:
        """
        Apply layer freezing strategy to model.

        Args:
            model: YOLO model instance
            strategy: Freezing strategy ('backbone', 'partial', 'none')
        """
        # Legacy path: kept for compatibility. Training now uses Ultralytics
        # native `freeze` argument in utils/training.py as single source of truth.
        if strategy == "none" or strategy is None:
            print("No layers frozen - full fine-tuning")
            return

        freeze_layers = self.model_config.get("freeze_layers", 10)

        if strategy == "backbone":
            # Freeze backbone (first N layers)
            print(f"Freezing backbone (first {freeze_layers} layers)")
            model.model.model[:freeze_layers].requires_grad_(False)

        elif strategy == "partial":
            # Freeze specific number of layers
            print(f"Freezing first {freeze_layers} layers (partial)")
            for i, layer in enumerate(model.model.model):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        self._print_frozen_status(model)

    def _print_frozen_status(self, model: YOLO) -> None:
        """Print which layers are frozen/trainable."""
        trainable = 0
        frozen = 0
        for param in model.model.parameters():
            if param.requires_grad:
                trainable += param.numel()
            else:
                frozen += param.numel()

        total = trainable + frozen
        print(f"  Trainable params: {trainable:,} ({100 * trainable / total:.1f}%)")
        print(f"  Frozen params: {frozen:,} ({100 * frozen / total:.1f}%)")

    def print_model_info(self, model: YOLO) -> None:
        """Print model architecture information."""
        print("\n" + "=" * 50)
        print("Model Information")
        print("=" * 50)

        # Count parameters
        total_params, trainable_params = self.count_parameters(model)

        print(f"  Model: {self.model_config.get('name', 'unknown')}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
        print("=" * 50 + "\n")

    def count_parameters(self, model: YOLO) -> Tuple[int, int]:
        """
        Count total and trainable parameters.

        Args:
            model: YOLO model

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(
            p.numel() for p in model.model.parameters() if p.requires_grad
        )
        return total_params, trainable_params

    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """Return dictionary of available YOLO models."""
        return ModelLoader.YOLO_VARIANTS.copy()

    @staticmethod
    def get_device() -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple MPS")
        else:
            device = "cpu"
            print("Using CPU")
        return device
