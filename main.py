"""
PCB Defect Detection - Main Entry Point
========================================
Train and evaluate YOLO models for PCB surface defect detection.

Usage:
    python main.py                                    # Train with defaults
    python main.py --config configs/config.yaml       # Custom config
    python main.py --mode evaluate --weights best.pt  # Evaluate model
    python main.py --mode predict --weights best.pt --source image.jpg
    python main.py --skip-setup                       # Skip auto-install
"""

import importlib.util
import subprocess
import sys
from pathlib import Path


def ensure_dependencies():
    """Auto-install requirements if missing."""
    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        return

    # Check if ultralytics is installed (key dependency)
    if importlib.util.find_spec("ultralytics") is not None:
        return  # Already installed

    print("📦 Installing dependencies...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "-q"]
        )
        print("✅ Dependencies installed!\n")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Failed to install dependencies: {e}")
        print("Run manually: pip install -r requirements.txt")


# Auto-install before importing dependencies
if "--skip-setup" not in sys.argv:
    ensure_dependencies()

import argparse  # noqa: E402
import yaml  # noqa: E402

import wandb  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from model.model_loader import ModelLoader  # noqa: E402
from utils.training import YOLOTrainer  # noqa: E402
from utils.evaluation import YOLOEvaluator  # noqa: E402
from utils.data_loader import DatasetManager  # noqa: E402
from utils.plots import TrainingPlotter  # noqa: E402


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from {config_path}")
    return config


def log_training_visuals(results) -> None:
    """Log custom plots and key images to WandB after training."""
    if results is None:
        return
    if wandb.run is None:
        print("Wandb run not active, skipping custom plot logging.")
        return

    save_dir = Path(results.save_dir)
    plotter = TrainingPlotter()
    visuals = {}

    results_csv = save_dir / "results.csv"
    try:
        loss_fig = plotter.plot_losses_from_results(str(results_csv))
        if loss_fig:
            visuals["plots/train_val_losses"] = wandb.Image(loss_fig)
            plotter.close_figure(loss_fig)
    except Exception as e:
        print(f"Could not create loss plot: {e}")

    try:
        metrics_fig = plotter.plot_metrics_from_results(str(results_csv))
        if metrics_fig:
            visuals["plots/val_metrics"] = wandb.Image(metrics_fig)
            plotter.close_figure(metrics_fig)
    except Exception as e:
        print(f"Could not create metrics plot: {e}")

    built_in_plots = {
        "results": save_dir / "results.png",
        "PR_curve": save_dir / "PR_curve.png",
        "F1_curve": save_dir / "F1_curve.png",
        "P_curve": save_dir / "P_curve.png",
        "R_curve": save_dir / "R_curve.png",
        "confusion_matrix": save_dir / "confusion_matrix.png",
        "confusion_matrix_normalized": save_dir / "confusion_matrix_normalized.png",
        "labels": save_dir / "labels.jpg",
        "labels_correlogram": save_dir / "labels_correlogram.jpg",
        "val_batch0_pred": save_dir / "val_batch0_pred.jpg",
    }

    for name, path in built_in_plots.items():
        if path.exists():
            visuals[f"ultralytics/{name}"] = wandb.Image(str(path))

    if visuals:
        wandb.log(visuals)
        print(f"Logged {len(visuals)} custom plots to wandb.")


def train(args, config: dict) -> None:
    """Run training pipeline with automatic test evaluation."""
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
    if args.freeze is not None:
        config["model"]["freeze_layers"] = args.freeze
    if args.name:
        config["output"]["name"] = args.name
        config["wandb"]["run_name"] = args.name
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
    log_training_visuals(results)

    # Run evaluation on test set if it exists
    test_ratio = config.get("data", {}).get("split", {}).get("test_ratio", 0)
    if test_ratio > 0 and results is not None:
        print("\n" + "=" * 60)
        print("  Running Final Evaluation on Test Set")
        print("=" * 60)

        best_weights = f"{results.save_dir}/weights/best.pt"
        evaluator = YOLOEvaluator(model_path=best_weights)

        # Evaluate on test set
        test_metrics = evaluator.evaluate(
            data_yaml=data_yaml_path,
            split="test",
            project=str(results.save_dir),
            name="test_evaluation",
            analyze_errors=True,
        )

        # Log test metrics to the active WandB run with test/ prefix
        # Metrics are in 0-1 scale (YOLO format) - multiply by 100 to compare with COCO-style
        wandb_test_metrics = {
            "test/mAP50": test_metrics.get("mAP50", 0),
            "test/mAP50-95": test_metrics.get("mAP50-95", 0),
            "test/precision": test_metrics.get("precision", 0),
            "test/recall": test_metrics.get("recall", 0),
        }
        # Add per-class AP50 metrics
        for key, value in test_metrics.items():
            if key.startswith(("AP50_", "AP50-95_", "Precision_", "Recall_")):
                wandb_test_metrics[f"test/{key}"] = value
        
        if wandb.run is not None:
            wandb.log(wandb_test_metrics)
            wandb.summary.update(wandb_test_metrics)
            print(f"\n✓ Test metrics logged to WandB:")
        else:
            print(f"\n⚠ WandB run not active, test metrics not logged to WandB:")
        
        print(f"    test/mAP50: {test_metrics.get('mAP50', 0):.4f} (×100 = {test_metrics.get('mAP50', 0)*100:.2f}%)")
        print(f"    test/mAP50-95: {test_metrics.get('mAP50-95', 0):.4f} (×100 = {test_metrics.get('mAP50-95', 0)*100:.2f}%)")

        print(
            f"\n✓ Test evaluation complete. Results saved to: {results.save_dir}/test_evaluation/"
        )

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
        project="runs/evaluate",
        name=args.name or "eval",
        analyze_errors=True,
    )

    return metrics


def predict(args, config: dict) -> None:
    """Run inference on images using YOLO directly."""
    print("\n" + "=" * 60)
    print("  PCB Defect Detection - Inference")
    print("=" * 60)

    if not args.weights:
        print("❌ Please provide --weights path to trained model")
        return

    if not args.source:
        print("❌ Please provide --source path to image(s)")
        return

    # Use YOLO directly for prediction (it handles visualization)
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=True,
        project=args.output or "runs/predict",
        show=args.show,
    )

    # Print summary
    total_detections = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
    print(f"\n✓ Processed {len(results)} images, found {total_detections} defects")


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
        "--freeze",
        type=int,
        default=None,
        help="Number of layers to freeze (0=none, 10=backbone)",
    )
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
    parser.add_argument(
        "--skip-setup", action="store_true", help="Skip auto-install of dependencies"
    )

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
