"""
Faster R-CNN Training with Detectron2
"""
# Usage: python train_frcnn.py --data-path /work/Data_COCO_unsplit --epochs 50

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import wandb
import yaml

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger

setup_logger()

CLASS_NAMES = ["SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO"]


def load_wandb_config(config_path="configs/config.yaml"):
    """Load WandB settings from existing config."""
    if Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("wandb", {})
    return {}


def stratified_split_coco(annotations_path, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split COCO annotations preserving class balance."""
    random.seed(seed)

    with open(annotations_path) as f:
        data = json.load(f)

    # Group images by class composition
    img_to_anns = defaultdict(list)
    for ann in data["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    class_groups = defaultdict(list)
    for img in data["images"]:
        cats = tuple(
            sorted(set(a["category_id"] for a in img_to_anns.get(img["id"], [])))
        ) or (-1,)
        class_groups[cats].append(img)

    # Split each group
    train_imgs, val_imgs, test_imgs = [], [], []
    for imgs in class_groups.values():
        random.shuffle(imgs)
        n_test = int(len(imgs) * test_ratio)
        n_val = int(len(imgs) * val_ratio)
        test_imgs.extend(imgs[:n_test])
        val_imgs.extend(imgs[n_test : n_test + n_val])
        train_imgs.extend(imgs[n_test + n_val :])

    # Build split datasets with unique annotation IDs
    def make_split(images):
        ids = {img["id"] for img in images}
        anns = [a.copy() for a in data["annotations"] if a["image_id"] in ids]
        # Re-index annotation IDs to ensure uniqueness
        for i, ann in enumerate(anns, start=1):
            ann["id"] = i
        return {
            "info": data.get("info", {"description": "PCB Defect Dataset"}),
            "licenses": data.get("licenses", []),
            "images": images,
            "annotations": anns,
            "categories": data["categories"],
        }

    print(f"Split: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
    return (
        make_split(train_imgs),
        make_split(val_imgs),
        make_split(test_imgs) if test_imgs else None,
    )


def prepare_data(data_path, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Prepare COCO data - handles unsplit or pre-split formats."""
    data_path = Path(data_path)

    # Pre-split format (no test set in COCO 2017 format)
    if (data_path / "annotations" / "instances_train2017.json").exists():
        train_json = data_path / "annotations" / "instances_train2017.json"
        val_json = data_path / "annotations" / "instances_val2017.json"
        with open(train_json) as f:
            n_train = len(json.load(f)["images"])
        return train_json, val_json, None, data_path / "train2017", n_train

    # Unsplit format, does a stratified split
    ann_file = data_path / "annotations.json"
    images_dir = data_path / "images"

    train_data, val_data, test_data = stratified_split_coco(
        ann_file, val_ratio, test_ratio, seed
    )

    # Save split jsons
    out_dir = data_path.parent / "PCB_Data_COCO"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_json, val_json = out_dir / "train.json", out_dir / "val.json"
    with open(train_json, "w") as f:
        json.dump(train_data, f)
    with open(val_json, "w") as f:
        json.dump(val_data, f)

    test_json = None
    if test_data:
        test_json = out_dir / "test.json"
        with open(test_json, "w") as f:
            json.dump(test_data, f)

    return train_json, val_json, test_json, images_dir, len(train_data["images"])


class WandbHook(HookBase):
    """Log training metrics to WandB (matches YOLO logging style)."""

    def __init__(self, period=20, max_iter=None):
        self.period = period
        self.max_iter = max_iter
        self._start_time = None

    def before_train(self):
        self._start_time = time.time()

    def after_step(self):
        if (self.trainer.iter + 1) % self.period != 0:
            return

        storage = self.trainer.storage
        metrics = {}

        # Training losses (like YOLO's train/box_loss, train/cls_loss, train/dfl_loss)
        loss_mapping = {
            "total_loss": "train/total_loss",
            "loss_cls": "train/cls_loss",  # Like YOLO's train/cls_loss
            "loss_box_reg": "train/box_loss",  # Like YOLO's train/box_loss
            "loss_rpn_cls": "train/rpn_cls_loss",
            "loss_rpn_loc": "train/rpn_box_loss",
        }
        for key, wandb_key in loss_mapping.items():
            if key in storage._history:
                try:
                    val, _ = storage.history(key).median(20)
                    metrics[wandb_key] = val
                except Exception:
                    pass

        # Learning rate (like YOLO's lr/pg0, lr/pg1, lr/pg2)
        if "lr" in storage._history:
            try:
                lr_val, _ = storage.history("lr").latest()
                metrics["lr/pg0"] = lr_val
                metrics["lr/pg1"] = lr_val
                metrics["lr/pg2"] = lr_val
            except Exception:
                pass

        # Training progress
        if self.max_iter:
            progress = (self.trainer.iter + 1) / self.max_iter
            metrics["train/progress"] = progress

            # Estimated time remaining
            if self._start_time:
                elapsed = time.time() - self._start_time
                if progress > 0:
                    eta_seconds = (elapsed / progress) * (1 - progress)
                    metrics["train/eta_minutes"] = eta_seconds / 60

        # Iteration timing
        if "time" in storage._history:
            try:
                iter_time, _ = storage.history("time").latest()
                metrics["train/iter_time"] = iter_time
            except Exception:
                pass

        if wandb.run and metrics:
            wandb.log(metrics, step=self.trainer.iter)


class WandbValHook(HookBase):
    """Log validation metrics to WandB after each eval (matches YOLO's per-epoch val logging)."""

    def __init__(self, eval_period, cfg):
        self.eval_period = eval_period
        self.cfg = cfg
        self._last_eval_iter = -1
        self._epoch = 0

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self.eval_period != 0 or next_iter == self._last_eval_iter:
            return
        self._last_eval_iter = next_iter
        self._epoch += 1

        # Get validation results from storage (logged by EvalHook)
        storage = self.trainer.storage
        metrics = {}

        # Main COCO metrics mapped to YOLO-style names
        # COCO returns 0-100 scale, normalize to 0-1 to match YOLO format
        metric_mapping = {
            "bbox/AP": "metrics/mAP50-95(B)",
            "bbox/AP50": "metrics/mAP50(B)",
            "bbox/AP75": "val/mAP75",
            "bbox/APs": "val/mAP_small",  # Small objects
            "bbox/APm": "val/mAP_medium",  # Medium objects
            "bbox/APl": "val/mAP_large",  # Large objects
        }

        for key, wandb_key in metric_mapping.items():
            full_key = f"pcb_val/{key}" if f"pcb_val/{key}" in storage._history else key
            if full_key in storage._history:
                try:
                    val, _ = storage.history(full_key).latest()
                    # Normalize from 0-100 to 0-1 scale to match YOLO
                    metrics[wandb_key] = val / 100.0
                except Exception:
                    pass

        # Per-class AP50 metrics (like YOLO's val/AP50_MB, val/AP50_CS, etc.)
        for i, cls_name in enumerate(CLASS_NAMES):
            key = f"bbox/AP50-{cls_name}"
            full_key = f"pcb_val/{key}" if f"pcb_val/{key}" in storage._history else key
            if full_key in storage._history:
                try:
                    val, _ = storage.history(full_key).latest()
                    metrics[f"val/AP50_{cls_name}"] = val / 100.0
                except Exception:
                    pass

        # Also log validation losses if available
        val_loss_keys = ["val_total_loss", "val_loss_cls", "val_loss_box_reg"]
        for key in val_loss_keys:
            if key in storage._history:
                try:
                    val, _ = storage.history(key).latest()
                    metrics[f"val/{key.replace('val_', '')}"] = val
                except Exception:
                    pass

        # Log epoch number for x-axis alignment with YOLO
        metrics["epoch"] = self._epoch

        if wandb.run and metrics:
            wandb.log(metrics, step=self.trainer.iter)


class TimeLimitHook(HookBase):
    """Stop training after a time limit (for fair comparison with YOLO)."""

    def __init__(self, max_minutes):
        self.max_seconds = max_minutes * 60
        self.start_time = None

    def before_train(self):
        self.start_time = time.time()

    def after_step(self):
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_seconds:
            elapsed_min = elapsed / 60
            print(f"\n⏱️ Time limit reached ({elapsed_min:.1f} min). Stopping training.")
            raise StopIteration  # Detectron2 catches this to stop training


class FRCNNTrainer(DefaultTrainer):
    _max_time_minutes = None  # Class variable, set BEFORE instantiation
    _max_iter = None  # For progress tracking

    def build_hooks(self):
        hooks = super().build_hooks()
        # Pass max_iter for progress tracking
        hooks.append(WandbHook(period=20, max_iter=FRCNNTrainer._max_iter))
        if self.cfg.TEST.EVAL_PERIOD > 0:
            hooks.append(WandbValHook(self.cfg.TEST.EVAL_PERIOD, self.cfg))
        if FRCNNTrainer._max_time_minutes:
            hooks.append(TimeLimitHook(FRCNNTrainer._max_time_minutes))
        return hooks

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(
            dataset_name,
            output_dir=os.path.join(cfg.OUTPUT_DIR, "eval"),
            tasks=("bbox",),  # Ensure per-class metrics are computed
        )


def train(args):
    print("\n" + "=" * 60)
    print("  Faster R-CNN Training (Detectron2)")
    print("=" * 60)

    # Prepare data
    train_json, val_json, test_json, images_dir, n_train = prepare_data(
        args.data_path, args.val_ratio, args.test_ratio, args.seed
    )

    # Register datasets
    for name in ["pcb_train", "pcb_val", "pcb_test"]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)

    register_coco_instances("pcb_train", {}, str(train_json), str(images_dir))
    register_coco_instances("pcb_val", {}, str(val_json), str(images_dir))
    MetadataCatalog.get("pcb_train").thing_classes = CLASS_NAMES
    MetadataCatalog.get("pcb_val").thing_classes = CLASS_NAMES

    if test_json:
        register_coco_instances("pcb_test", {}, str(test_json), str(images_dir))
        MetadataCatalog.get("pcb_test").thing_classes = CLASS_NAMES

    # Calculate iterations
    iters_per_epoch = max(1, n_train // args.batch)
    max_iter = iters_per_epoch * args.epochs

    print(f"\n  Train images: {n_train}, Epochs: {args.epochs}, Iters: {max_iter}")

    # Setup config
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.DATASETS.TRAIN = ("pcb_train",)
    cfg.DATASETS.TEST = ("pcb_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.SOLVER.IMS_PER_BATCH = args.batch
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = (int(max_iter * 0.7), int(max_iter * 0.9))
    cfg.SOLVER.CHECKPOINT_PERIOD = iters_per_epoch * 5
    cfg.SOLVER.WARMUP_ITERS = min(1000, max_iter // 10)
    # Note: Using Detectron2 defaults for momentum (0.9) and weight_decay (0.0001)
    # These are the recommended values for Faster R-CNN, tuned for this architecture
    cfg.TEST.EVAL_PERIOD = iters_per_epoch
    cfg.OUTPUT_DIR = f"runs/frcnn/{args.name}"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Init WandB with comprehensive config (like YOLO)
    wandb_cfg = load_wandb_config()
    wandb.init(
        project=wandb_cfg.get("project", "PCB_Defect_Detection"),
        entity=wandb_cfg.get("entity"),
        name=args.name,
        config={
            # Model config
            "model": "faster_rcnn_R_50_FPN",
            "backbone": "ResNet50-FPN",
            "pretrained": True,
            # Training config (like YOLO's training.*)
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": 800,  # Detectron2 default short edge
            "lr": args.lr,
            "momentum": 0.9,  # Detectron2 default
            "weight_decay": 0.0001,  # Detectron2 default
            "warmup_iters": cfg.SOLVER.WARMUP_ITERS,
            "max_iter": max_iter,
            # Data config
            "num_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "train_images": n_train,
        },
        tags=["faster-rcnn", "detectron2", "pcb"],
    )

    # Log model info (like YOLO's model/GFLOPs, model/parameters)
    if wandb.run:
        # Rough parameter count for Faster R-CNN R50-FPN
        wandb.log(
            {
                "model/parameters": 41_000_000,  # ~41M for R50-FPN
                "model/backbone": "ResNet50",
                "model/neck": "FPN",
            }
        )

    # Train
    start_time = time.time()
    FRCNNTrainer._max_iter = max_iter  # For progress tracking
    if args.max_time:
        FRCNNTrainer._max_time_minutes = args.max_time  # Set BEFORE instantiation
        print(f"  Time limit: {args.max_time} minutes")
    trainer = FRCNNTrainer(cfg)
    trainer.resume_or_load(resume=False)
    try:
        trainer.train()
    except StopIteration:
        pass  # Time limit reached
    train_time = (time.time() - start_time) / 60
    print(f"\n  Training time: {train_time:.1f} minutes")
    if wandb.run:
        wandb.log({"train/time_minutes": train_time})

    # Final Evaluation on TEST set (like YOLO's test set evaluation)
    eval_dataset = "pcb_test" if test_json else "pcb_val"
    eval_prefix = "test" if test_json else "val"
    print(f"\n  Running final evaluation on {eval_dataset}...")

    evaluator = COCOEvaluator(
        eval_dataset,
        output_dir=os.path.join(cfg.OUTPUT_DIR, "eval"),
        tasks=("bbox",),
    )
    results = inference_on_dataset(
        trainer.model, build_detection_test_loader(cfg, eval_dataset), evaluator
    )

    if "bbox" in results:
        bbox = results["bbox"]

        # COCO returns metrics in 0-100 scale, normalize to 0-1 to match YOLO format
        # This makes direct comparison possible: both models log in 0-1 scale
        ap50 = bbox.get("AP50", 0) / 100.0
        ap = bbox.get("AP", 0) / 100.0
        ap75 = bbox.get("AP75", 0) / 100.0

        metrics = {
            # YOLO-compatible names and scale for direct comparison
            "metrics/mAP50(B)": ap50,
            "metrics/mAP50-95(B)": ap,
            # Also log with test/val prefix (like YOLO's test/mAP50)
            f"{eval_prefix}/mAP50": ap50,
            f"{eval_prefix}/mAP50-95": ap,
            f"{eval_prefix}/mAP75": ap75,
            # Size-based metrics (small/medium/large objects)
            f"{eval_prefix}/mAP_small": bbox.get("APs", 0) / 100.0,
            f"{eval_prefix}/mAP_medium": bbox.get("APm", 0) / 100.0,
            f"{eval_prefix}/mAP_large": bbox.get("APl", 0) / 100.0,
            # Final metrics for summary
            "final/mAP50": ap50,
            "final/mAP50-95": ap,
        }

        print(f"  mAP50: {ap50:.4f} ({bbox.get('AP50', 0):.2f}%)")
        print(f"  mAP50-95: {ap:.4f} ({bbox.get('AP', 0):.2f}%)")
        print(f"  mAP75: {ap75:.4f} ({bbox.get('AP75', 0):.2f}%)")
        print(f"  AP (small): {bbox.get('APs', 0):.2f}%")
        print(f"  AP (medium): {bbox.get('APm', 0):.2f}%")
        print(f"  AP (large): {bbox.get('APl', 0):.2f}%")

        # Log per-class AP if available (like YOLO's val/AP50_MB, val/AP50_CS)
        if hasattr(evaluator, "_coco_eval") and evaluator._coco_eval:
            try:
                coco_eval = evaluator._coco_eval["bbox"]
                # Per-class AP at IoU=0.5
                precisions = coco_eval.eval["precision"]
                # precision has shape (iou_thresholds, recall, classes, areas, max_dets)
                # IoU=0.5 is index 0, all recalls, all areas (index 0), max_det 100
                for i, cls_name in enumerate(CLASS_NAMES):
                    if i < precisions.shape[2]:
                        ap50_cls = precisions[0, :, i, 0, -1].mean() * 100
                        if ap50_cls >= 0:
                            metrics[f"{eval_prefix}/AP50_{cls_name}"] = ap50_cls / 100.0
                            print(f"  AP50 {cls_name}: {ap50_cls:.2f}%")
            except Exception as e:
                print(f"  (Per-class AP not available: {e})")

        wandb.log(metrics)
        wandb.summary.update(metrics)

    wandb.finish()
    print(f"\n END, Weights saved to {cfg.OUTPUT_DIR}/model_final.pth")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN")
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to COCO data"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--name", type=str, default="frcnn_train")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-time",
        type=int,
        default=None,
        help="Max training time in minutes (for fair comparison)",
    )

    train(parser.parse_args())


if __name__ == "__main__":
    main()
