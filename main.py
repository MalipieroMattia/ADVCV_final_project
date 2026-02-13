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
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def ensure_dependencies():
    """Auto-install requirements if missing."""
    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        return

    # Install requirements only if key modules are missing.
    required_modules = ["ultralytics", "pycocotools"]
    missing_modules = [
        m for m in required_modules if importlib.util.find_spec(m) is None
    ]
    if not missing_modules:
        return

    print("Installing dependencies...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "-q"]
        )
        print("Dependencies installed!\n")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        print("Run manually: pip install -r requirements.txt")


# Auto-install before importing dependencies
if "--skip-setup" not in sys.argv:
    ensure_dependencies()

import argparse  # noqa: E402
import importlib.metadata as importlib_metadata  # noqa: E402
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


def count_parameters(model) -> tuple[int, int]:
    """Return (total_params, trainable_params) for a YOLO model."""
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(
        p.numel() for p in model.model.parameters() if p.requires_grad
    )
    return total_params, trainable_params


def ensure_wandb_run(
    config: dict, run_id: str | None = None, run_name: str | None = None
) -> bool:
    """
    Ensure a W&B run exists for this process.

    Returns:
        True if this function created the run, False otherwise.
    """
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", True):
        return False
    if wandb.run is not None:
        return False

    init_kwargs = {
        "project": wandb_cfg.get("project", "PCB_Defect_Detection"),
        "entity": wandb_cfg.get("entity"),
        "name": run_name or wandb_cfg.get("run_name"),
        "tags": wandb_cfg.get("tags", ["yolov8", "pcb"]),
        "mode": wandb_cfg.get("mode", "online"),
        "config": {
            "model": config.get("model", {}),
            "training": config.get("training", {}),
            "augmentation": config.get("augmentation", {}),
            "data": config.get("data", {}),
        },
    }
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"

    try:
        wandb.init(**init_kwargs)
        print(f"W&B run initialized: {wandb.run.name}")
        return True
    except Exception as e:
        print(f"Could not initialize W&B run explicitly: {e}")
        return False


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


def _get_git_sha() -> str:
    """Return current git commit SHA if available."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=Path(__file__).parent,
        ).strip()
        return sha
    except Exception:
        return "unknown"


def _get_package_version(package_name: str) -> str:
    """Return installed package version, or 'not_installed'."""
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"
    except Exception:
        return "unknown"


def log_run_artifact(
    results,
    config_path: str,
    data_yaml_path: str,
    config: dict,
    test_metrics: dict | None = None,
    elapsed_minutes: float | None = None,
) -> None:
    """
    Log a full run bundle as a W&B artifact.

    Includes run outputs directory, config, data.yaml, and run manifest.
    """
    if results is None:
        return
    if wandb.run is None:
        print("Wandb run not active, skipping artifact upload.")
        return

    save_dir = Path(results.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "wandb_run_id": wandb.run.id,
        "wandb_run_name": wandb.run.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _get_git_sha(),
        "ultralytics_version": _get_package_version("ultralytics"),
        "pycocotools_version": _get_package_version("pycocotools"),
        "config_path": str(Path(config_path).resolve()),
        "data_yaml_path": str(Path(data_yaml_path).resolve()),
        "save_dir": str(save_dir.resolve()),
        "model_name": config.get("model", {}).get("name", "unknown"),
        "freeze_layers": config.get("model", {}).get("freeze_layers", 0),
        "epochs": config.get("training", {}).get("epochs", None),
        "imgsz": config.get("training", {}).get("imgsz", None),
        "batch_size": config.get("training", {}).get("batch_size", None),
        "elapsed_minutes": elapsed_minutes,
        "test_metrics": test_metrics or {},
    }
    manifest_path = save_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    artifact = wandb.Artifact(
        name=f"run_bundle_{wandb.run.id}",
        type="training-run",
        description="Training and evaluation outputs bundle",
        metadata={
            "run_id": wandb.run.id,
            "model": manifest["model_name"],
            "freeze_layers": manifest["freeze_layers"],
            "git_sha": manifest["git_sha"],
        },
    )

    artifact.add_dir(str(save_dir), name="run_outputs")

    cfg_path = Path(config_path)
    if cfg_path.exists():
        artifact.add_file(str(cfg_path), name="config_used.yaml")

    data_cfg = Path(data_yaml_path)
    if data_cfg.exists():
        artifact.add_file(str(data_cfg), name="data_used.yaml")

    wandb.log_artifact(artifact, aliases=["latest", wandb.run.id])
    print(f"Uploaded W&B artifact: run_bundle_{wandb.run.id}")


def keep_best_weight_only(results) -> None:
    """Keep only best.pt in the weights directory."""
    if results is None:
        return

    weights_dir = Path(results.save_dir) / "weights"
    if not weights_dir.exists():
        return

    best_path = weights_dir / "best.pt"
    if not best_path.exists():
        print(f"best.pt not found in {weights_dir}, skipping weight cleanup")
        return

    removed = 0
    for pt_file in weights_dir.glob("*.pt"):
        if pt_file.name == "best.pt":
            continue
        try:
            pt_file.unlink()
            removed += 1
        except Exception as e:
            print(f"Could not remove {pt_file}: {e}")

    if removed:
        print(f"Kept only best.pt in {weights_dir} (removed {removed} files)")


def _count_gt_instances(data_yaml_path: str, split: str = "test") -> int:
    """Count GT annotations for a split from data.yaml."""
    with open(data_yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg["path"])
    split_rel = Path(data_cfg.get(split, f"images/{split}"))
    split_name = split_rel.name
    labels_dir = base_path / "labels" / split_name

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found for split '{split}': {labels_dir}")

    total = 0
    for txt in labels_dir.glob("*.txt"):
        with open(txt, "r") as f:
            total += sum(1 for ln in f if ln.strip())
    return total


def run_p2_conf_sweep(
    *,
    best_weights: str,
    data_yaml_path: str,
    run_dir: str | Path,
    split: str = "test",
    iou_threshold: float = 0.5,
    conf_values: list[float] | None = None,
) -> Path:
    """
    Sweep confidence thresholds for P2 runs and save error-based PR/F1 summary.

    Metrics are derived from the same TP/FP/FN logic used by the project's error analysis:
    - wrong_class counts as both one FP and one FN
    """
    if conf_values is None:
        conf_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]

    gt_total = _count_gt_instances(data_yaml_path=data_yaml_path, split=split)
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{split}_conf_sweep.csv"

    rows = []
    print("\n" + "=" * 60)
    print(f"  Confidence Sweep ({split})")
    print("=" * 60)
    print(f"GT instances: {gt_total}")
    print(f"IoU for matching: {iou_threshold}")
    print(f"Thresholds: {conf_values}")

    for conf in conf_values:
        evaluator = YOLOEvaluator(
            model_path=best_weights,
            conf_threshold=float(conf),
            iou_threshold=float(iou_threshold),
        )
        errors = evaluator._analyze_errors(data_yaml=data_yaml_path, split=split)

        n_fp = sum(1 for e in errors if e["error_type"] == "false_positive")
        n_missed = sum(1 for e in errors if e["error_type"] == "missed")
        n_wrong = sum(1 for e in errors if e["error_type"] == "wrong_class")

        tp = max(gt_total - n_missed - n_wrong, 0)
        fp = n_fp + n_wrong
        fn = n_missed + n_wrong

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        row = {
            "conf": float(conf),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "false_positive": int(n_fp),
            "missed": int(n_missed),
            "wrong_class": int(n_wrong),
            "precision_err_based": float(precision),
            "recall_err_based": float(recall),
            "f1_err_based": float(f1),
        }
        rows.append(row)
        print(
            f"  conf={conf:.2f} -> "
            f"P={precision:.4f}, R={recall:.4f}, F1={f1:.4f} "
            f"(FP={n_fp}, Missed={n_missed}, Wrong={n_wrong})"
        )

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved confidence sweep CSV to: {out_csv}")

    if wandb.run is not None and rows:
        best_row = max(rows, key=lambda x: x["f1_err_based"])
        wandb.log(
            {
                f"{split}/conf_sweep/best_conf": best_row["conf"],
                f"{split}/conf_sweep/best_f1_err_based": best_row["f1_err_based"],
                f"{split}/conf_sweep/best_precision_err_based": best_row["precision_err_based"],
                f"{split}/conf_sweep/best_recall_err_based": best_row["recall_err_based"],
            }
        )
        wandb.summary.update(
            {
                f"{split}/conf_sweep/best_conf": best_row["conf"],
                f"{split}/conf_sweep/best_f1_err_based": best_row["f1_err_based"],
            }
        )

    return out_csv


def _pick_best_conf_from_sweep_csv(sweep_csv: str | Path) -> float | None:
    """Pick confidence with highest error-based F1 from a sweep CSV."""
    sweep_csv = Path(sweep_csv)
    if not sweep_csv.exists():
        return None

    best_conf = None
    best_f1 = -1.0

    with open(sweep_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                conf = float(row.get("conf", "nan"))
                f1 = float(row.get("f1_err_based", "nan"))
            except (TypeError, ValueError):
                continue

            if f1 > best_f1:
                best_f1 = f1
                best_conf = conf

    return best_conf


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
        print("\nTEST RUN MODE: training for 1 epoch only\n")

    created_wandb_run = ensure_wandb_run(config)
    wandb_run_id = wandb.run.id if wandb.run is not None else None
    wandb_run_name = wandb.run.name if wandb.run is not None else None

    results = None
    model = None
    data_yaml_path = None
    test_metrics_best = None
    test_metrics_last = None
    elapsed_minutes = None
    test_metrics_conf_selected = None
    selected_conf_for_test = None

    try:
        device = ModelLoader.get_device()
        config["training"]["device"] = device

        data_manager = DatasetManager(config)
        is_valid, _counts = data_manager.verify_dataset()

        if not is_valid:
            print("\nDataset not found or incomplete")
            return

        data_yaml_path = data_manager.create_data_yaml()

        model_loader = ModelLoader(config)
        model = model_loader.load_model(checkpoint_path=args.resume)

        trainer = YOLOTrainer(model, config, data_yaml_path)
        start_time = time.time()
        results = trainer.train()
        elapsed_minutes = (time.time() - start_time) / 60.0

        if wandb.run is None and wandb_run_id:
            ensure_wandb_run(config, run_id=wandb_run_id, run_name=wandb_run_name)

        log_training_visuals(results)

        if wandb.run is not None and model is not None:
            total_params, trainable_params = count_parameters(model)
            frozen_params = total_params - trainable_params
            trainable_pct = (
                (100.0 * trainable_params / total_params) if total_params else 0.0
            )

            wandb_metrics = {
                "train/time_minutes": elapsed_minutes,
                "model/trainable_parameters": trainable_params,
                "model/frozen_parameters": frozen_params,
                "model/trainable_pct": trainable_pct,
            }
            wandb.log(wandb_metrics)
            wandb.summary.update(wandb_metrics)

        # Run evaluation on test set if it exists
        test_ratio = config.get("data", {}).get("split", {}).get("test_ratio", 0)
        if test_ratio > 0 and results is not None:
            print("\n" + "=" * 60)
            print("  Running Final Evaluation on Test Set")
            print("=" * 60)

            weights_dir = Path(results.save_dir) / "weights"
            best_weights = str(weights_dir / "best.pt")
            last_weights = str(weights_dir / "last.pt")

            evaluator_best = YOLOEvaluator(model_path=best_weights)
            test_metrics_best = evaluator_best.evaluate(
                data_yaml=data_yaml_path,
                split="test",
                project=str(results.save_dir),
                name="test_evaluation",
                analyze_errors=True,
                compute_coco_metrics=True,
            )

            # Also evaluate last.pt for equal-budget comparison, when available.
            if Path(last_weights).exists():
                evaluator_last = YOLOEvaluator(model_path=last_weights)
                test_metrics_last = evaluator_last.evaluate(
                    data_yaml=data_yaml_path,
                    split="test",
                    project=str(results.save_dir),
                    name="test_evaluation_last",
                    analyze_errors=True,
                    compute_coco_metrics=True,
                )
            else:
                print(f"\nlast.pt not found at {last_weights}, skipping last-checkpoint evaluation")

            if wandb.run is None and wandb_run_id:
                ensure_wandb_run(config, run_id=wandb_run_id, run_name=wandb_run_name)

            # Keep `test/*` as best-checkpoint metrics for reporting consistency.
            wandb_test_metrics = {
                "test/mAP50": test_metrics_best.get("mAP50", 0),
                "test/mAP50-95": test_metrics_best.get("mAP50-95", 0),
                "test/precision": test_metrics_best.get("precision", 0),
                "test/recall": test_metrics_best.get("recall", 0),
            }
            for key, value in test_metrics_best.items():
                if key.startswith(
                    ("AP50_", "AP50-95_", "Precision_", "Recall_", "coco/")
                ):
                    wandb_test_metrics[f"test/{key}"] = value

            # Additional namespace for last-checkpoint test metrics.
            if test_metrics_last is not None:
                wandb_test_metrics.update(
                    {
                        "test_last/mAP50": test_metrics_last.get("mAP50", 0),
                        "test_last/mAP50-95": test_metrics_last.get("mAP50-95", 0),
                        "test_last/precision": test_metrics_last.get("precision", 0),
                        "test_last/recall": test_metrics_last.get("recall", 0),
                    }
                )
                for key, value in test_metrics_last.items():
                    if key.startswith(
                        ("AP50_", "AP50-95_", "Precision_", "Recall_", "coco/")
                    ):
                        wandb_test_metrics[f"test_last/{key}"] = value

            if wandb.run is not None:
                wandb.log(wandb_test_metrics)
                wandb.summary.update(wandb_test_metrics)
                print("\nTest metrics logged to WandB")
            else:
                print("\nWandB run not active, test metrics not logged")

            print(f"    best test/mAP50: {test_metrics_best.get('mAP50', 0):.4f}")
            print(f"    best test/mAP50-95: {test_metrics_best.get('mAP50-95', 0):.4f}")
            if test_metrics_last is not None:
                print(f"    last test/mAP50: {test_metrics_last.get('mAP50', 0):.4f}")
                print(f"    last test/mAP50-95: {test_metrics_last.get('mAP50-95', 0):.4f}")
            print(
                f"\nTest evaluation complete. Results saved to: {results.save_dir}/test_evaluation/"
            )

        keep_best_weight_only(results)

        # Optional post-train confidence sweep for P2 models.
        model_name = str(config.get("model", {}).get("name", ""))
        if (
            not args.test_run
            and results is not None
            and data_yaml_path is not None
            and model_name.endswith("-p2")
        ):
            sweep_cfg = config.get("analysis", {}).get("conf_sweep", {})
            sweep_enabled = sweep_cfg.get("enabled", True)

            if sweep_enabled:
                sweep_split = str(sweep_cfg.get("split", "test"))
                sweep_iou = float(sweep_cfg.get("iou", 0.5))
                sweep_values = sweep_cfg.get(
                    "values", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
                )

                # Run in save_dir root so CSV is included in run artifact.
                best_weights = f"{results.save_dir}/weights/best.pt"
                try:
                    sweep_csv_path = run_p2_conf_sweep(
                        best_weights=best_weights,
                        data_yaml_path=data_yaml_path,
                        run_dir=results.save_dir,
                        split=sweep_split,
                        iou_threshold=sweep_iou,
                        conf_values=[float(v) for v in sweep_values],
                    )

                    # If selection is on val, automatically run one final test eval at conf*.
                    if sweep_split.lower() == "val" and test_ratio > 0:
                        selected_conf_for_test = _pick_best_conf_from_sweep_csv(sweep_csv_path)
                        if selected_conf_for_test is not None:
                            print("\n" + "=" * 60)
                            print("  Final Test Evaluation at val-selected confidence")
                            print("=" * 60)
                            print(
                                f"Selected conf* from val sweep: {selected_conf_for_test:.3f}"
                            )

                            evaluator_conf = YOLOEvaluator(
                                model_path=best_weights,
                                conf_threshold=float(selected_conf_for_test),
                                iou_threshold=float(sweep_iou),
                            )
                            test_metrics_conf_selected = evaluator_conf.evaluate(
                                data_yaml=data_yaml_path,
                                split="test",
                                project=str(results.save_dir),
                                name="test_evaluation_conf_selected",
                                analyze_errors=True,
                                compute_coco_metrics=True,
                            )

                            if wandb.run is None and wandb_run_id:
                                ensure_wandb_run(
                                    config,
                                    run_id=wandb_run_id,
                                    run_name=wandb_run_name,
                                )

                            if wandb.run is not None:
                                log_selected = {
                                    "test_conf_selected/conf": float(
                                        selected_conf_for_test
                                    ),
                                    "test_conf_selected/mAP50": test_metrics_conf_selected.get(
                                        "mAP50", 0
                                    ),
                                    "test_conf_selected/mAP50-95": test_metrics_conf_selected.get(
                                        "mAP50-95", 0
                                    ),
                                    "test_conf_selected/precision": test_metrics_conf_selected.get(
                                        "precision", 0
                                    ),
                                    "test_conf_selected/recall": test_metrics_conf_selected.get(
                                        "recall", 0
                                    ),
                                }
                                for key, value in test_metrics_conf_selected.items():
                                    if key.startswith(
                                        (
                                            "AP50_",
                                            "AP50-95_",
                                            "Precision_",
                                            "Recall_",
                                            "coco/",
                                        )
                                    ):
                                        log_selected[f"test_conf_selected/{key}"] = value
                                wandb.log(log_selected)
                                wandb.summary.update(log_selected)

                            print(
                                f"test_conf_selected/mAP50: {test_metrics_conf_selected.get('mAP50', 0):.4f}"
                            )
                            print(
                                f"test_conf_selected/mAP50-95: {test_metrics_conf_selected.get('mAP50-95', 0):.4f}"
                            )
                        else:
                            print(
                                "Could not pick conf* from val sweep CSV, skipping selected-conf test evaluation"
                            )
                except Exception as e:
                    print(f"Confidence sweep failed (continuing): {e}")

        if wandb.run is None and wandb_run_id:
            ensure_wandb_run(config, run_id=wandb_run_id, run_name=wandb_run_name)

        log_run_artifact(
            results=results,
            config_path=args.config,
            data_yaml_path=data_yaml_path,
            config=config,
            test_metrics={
                "best": test_metrics_best or {},
                "last": test_metrics_last or {},
                "conf_selected": test_metrics_conf_selected or {},
                "selected_conf": selected_conf_for_test,
            },
            elapsed_minutes=elapsed_minutes,
        )

        return results
    finally:
        if created_wandb_run and wandb.run is not None:
            wandb.finish()


def evaluate(args, config: dict) -> None:
    """Run evaluation pipeline."""
    print("\n" + "=" * 60)
    print("PCB Defect Detection - Evaluation")
    print("=" * 60)

    if not args.weights:
        print("Please provide --weights path to trained model")
        return

    if args.data_path:
        config["data"]["root"] = args.data_path
    if args.name:
        config["wandb"]["run_name"] = args.name

    created_wandb_run = ensure_wandb_run(config)

    try:
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
            compute_coco_metrics=True,
        )

        return metrics
    finally:
        if created_wandb_run and wandb.run is not None:
            wandb.finish()


def predict(args, config: dict) -> None:
    """Run inference on images using YOLO directly."""
    print("\n" + "=" * 60)
    print("  PCB Defect Detection - Inference")
    print("=" * 60)

    if not args.weights:
        print("Please provide --weights path to trained model")
        return

    if not args.source:
        print("Please provide --source path to image(s)")
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
