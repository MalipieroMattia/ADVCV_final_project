"""
PCB Defect Detection - Main Entry Point
========================================
Train and evaluate YOLO models for PCB surface defect detection.

Usage:
    python main.py                                    # Train with defaults
    python main.py --config configs/config.yaml       # Custom config
    python main.py --mode evaluate --weights best.pt  # Evaluate model
    python main.py --mode predict --weights best.pt --source image.jpg
"""

import argparse
from pathlib import Path
import yaml

from model.model_loader import ModelLoader
from utils.training import YOLOTrainer
from utils.evaluation import YOLOEvaluator
from utils.data_loader import DatasetManager


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from {config_path}")
    return config


def train(args, config: dict) -> None:
    """Run training pipeline."""
    print("\n" + "=" * 60)
    print("  PCB Defect Detection - Training")
    print("=" * 60)

    # Override config with CLI arguments
    if args.data_path:
        config["data"]["root"] = args.data_path
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch:
        config["training"]["batch_size"] = args.batch
    if args.model:
        config["model"]["name"] = args.model
    if args.test_run:
        config["training"]["epochs"] = 1
        print("\n🧪 TEST RUN MODE: Training for 1 epoch only\n")

    device = ModelLoader.get_device()
    config["training"]["device"] = device

    data_manager = DatasetManager(config)
    is_valid, counts = data_manager.verify_dataset()

    if not is_valid:
        print("\n❌ Dataset not found or incomplete!")
        return

    data_yaml_path = data_manager.create_data_yaml()

    model_loader = ModelLoader(config)
    model = model_loader.load_model(checkpoint_path=args.resume)

    trainer = YOLOTrainer(model, config, data_yaml_path)
    results = trainer.train()

    return results


def evaluate(args, config: dict) -> None:
    """Run evaluation pipeline."""
    print("\n" + "=" * 60)
    print("  PCB Defect Detection - Evaluation")
    print("=" * 60)

    if not args.weights:
        print("❌ Please provide --weights path to trained model")
        return

    if args.data_path:
        config["data"]["root"] = args.data_path

    data_manager = DatasetManager(config)
    data_yaml_path = data_manager.create_data_yaml()

    evaluator = YOLOEvaluator(
        model_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    metrics = evaluator.evaluate(
        data_yaml=data_yaml_path,
        split=args.split,
        save_results=True,
        project="runs/evaluate",
        name=args.name or "eval",
    )

    return metrics


def predict(args, config: dict) -> None:
    """Run inference on images."""
    print("\n" + "=" * 60)
    print("  PCB Defect Detection - Inference")
    print("=" * 60)

    if not args.weights:
        print("❌ Please provide --weights path to trained model")
        return

    if not args.source:
        print("❌ Please provide --source path to image(s)")
        return

    evaluator = YOLOEvaluator(
        model_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    source = Path(args.source)

    if source.is_dir():
        evaluator.batch_predict(
            image_dir=str(source),
            output_dir=args.output,
        )
    else:
        annotated, detections = evaluator.predict_and_visualize(
            image_path=str(source),
            output_path=args.output,
            show=args.show,
        )
        print(f"\nDetected {len(detections)} defects:")
        for det in detections:
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PCB Defect Detection with YOLO")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "predict"],
        default="train",
        help="Mode: train, evaluate, or predict",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--data-path", type=str, default=None, help="Override data path"
    )
    parser.add_argument("--model", type=str, default=None, help="YOLO model to use")
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to trained weights"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--test-run", action="store_true", help="Quick test run with 1 epoch"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--source", type=str, default=None, help="Image or directory for prediction"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument(
        "--output", type=str, default=None, help="Output path for predictions"
    )
    parser.add_argument(
        "--show", action="store_true", help="Display prediction results"
    )
    parser.add_argument("--name", type=str, default=None, help="Run name for outputs")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.mode == "train":
        train(args, config)
    elif args.mode == "evaluate":
        evaluate(args, config)
    elif args.mode == "predict":
        predict(args, config)


if __name__ == "__main__":
    main()
