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

    # Pre-split format
    if (data_path / "annotations" / "instances_train2017.json").exists():
        train_json = data_path / "annotations" / "instances_train2017.json"
        val_json = data_path / "annotations" / "instances_val2017.json"
        with open(train_json) as f:
            n_train = len(json.load(f)["images"])
        return (
            train_json,
            val_json,
            data_path / "train2017",
            data_path / "val2017",
            n_train,
        )

    # Unsplit format, does a stratified split
    ann_file = data_path / "annotations.json"
    images_dir = data_path / "images"

    train_data, val_data, _ = stratified_split_coco(
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

    return train_json, val_json, images_dir, images_dir, len(train_data["images"])


class WandbHook(HookBase):
    """Log training metrics to WandB (matches YOLO logging style)."""

    def __init__(self, period=20):
        self.period = period

    def after_step(self):
        if (self.trainer.iter + 1) % self.period != 0:
            return

        storage = self.trainer.storage
        metrics = {}

        # Training losses (like YOLO's train/box_loss, train/cls_loss)
        for key in [
            "total_loss",
            "loss_cls",
            "loss_box_reg",
            "loss_rpn_cls",
            "loss_rpn_loc",
        ]:
            if key in storage._history:
                try:
                    metrics[f"train/{key}"], _ = storage.history(key).median(20)
                except Exception:
                    pass

        # Learning rate (like YOLO's lr/pg0)
        if "lr" in storage._history:
            try:
                metrics["train/lr"], _ = storage.history("lr").latest()
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

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self.eval_period != 0 or next_iter == self._last_eval_iter:
            return
        self._last_eval_iter = next_iter

        # Get validation results from storage (logged by EvalHook)
        storage = self.trainer.storage
        metrics = {}

        # COCO metrics (like YOLO's metrics/mAP50, metrics/mAP50-95)
        for key in ["bbox/AP", "bbox/AP50", "bbox/AP75"]:
            full_key = f"pcb_val/{key}" if f"pcb_val/{key}" in storage._history else key
            if full_key in storage._history:
                try:
                    val, _ = storage.history(full_key).latest()
                    # Map to YOLO-style names
                    name = key.replace("bbox/", "metrics/")
                    if name == "metrics/AP":
                        name = "metrics/mAP50-95"
                    elif name == "metrics/AP50":
                        name = "metrics/mAP50"
                    metrics[name] = val
                except Exception:
                    pass

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
    max_time_minutes = None  # Set before training

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(WandbHook())
        if self.cfg.TEST.EVAL_PERIOD > 0:
            hooks.append(WandbValHook(self.cfg.TEST.EVAL_PERIOD, self.cfg))
        if self.max_time_minutes:
            hooks.append(TimeLimitHook(self.max_time_minutes))
        return hooks

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(
            dataset_name, output_dir=os.path.join(cfg.OUTPUT_DIR, "eval")
        )


def train(args):
    print("\n" + "=" * 60)
    print("  Faster R-CNN Training (Detectron2)")
    print("=" * 60)

    # Prepare data
    train_json, val_json, train_imgs, val_imgs, n_train = prepare_data(
        args.data_path, args.val_ratio, args.test_ratio, args.seed
    )

    # Register datasets
    for name in ["pcb_train", "pcb_val"]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)

    register_coco_instances("pcb_train", {}, str(train_json), str(train_imgs))
    register_coco_instances("pcb_val", {}, str(val_json), str(val_imgs))
    MetadataCatalog.get("pcb_train").thing_classes = CLASS_NAMES
    MetadataCatalog.get("pcb_val").thing_classes = CLASS_NAMES

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
    cfg.TEST.EVAL_PERIOD = iters_per_epoch
    cfg.OUTPUT_DIR = f"runs/frcnn/{args.name}"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Init WandB
    wandb_cfg = load_wandb_config()
    wandb.init(
        project=wandb_cfg.get("project", "PCB_Defect_Detection"),
        entity=wandb_cfg.get("entity"),
        name=args.name,
        config={
            "model": "faster_rcnn_R_50_FPN",
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
        },
        tags=["faster-rcnn", "detectron2", "pcb"],
    )

    # Train
    start_time = time.time()
    trainer = FRCNNTrainer(cfg)
    if args.max_time:
        trainer.max_time_minutes = args.max_time
        print(f"  Time limit: {args.max_time} minutes")
    trainer.resume_or_load(resume=False)
    try:
        trainer.train()
    except StopIteration:
        pass  # Time limit reached
    train_time = (time.time() - start_time) / 60
    print(f"\n  Training time: {train_time:.1f} minutes")
    if wandb.run:
        wandb.log({"train/time_minutes": train_time})

    # Final Evaluation (like YOLO's test set evaluation)
    print("\n  Running final evaluation...")
    evaluator = COCOEvaluator(
        "pcb_val", output_dir=os.path.join(cfg.OUTPUT_DIR, "eval")
    )
    results = inference_on_dataset(
        trainer.model, build_detection_test_loader(cfg, "pcb_val"), evaluator
    )

    if "bbox" in results:
        bbox = results["bbox"]

        # Main metrics (like YOLO's mAP50, mAP50-95, precision, recall)
        metrics = {
            "val/mAP50-95": bbox.get("AP", 0),
            "val/mAP50": bbox.get("AP50", 0),
            "val/mAP75": bbox.get("AP75", 0),
        }

        # Per-class AP (like YOLO's AP50_SH, AP50_SP, etc.)
        for i, cls_name in enumerate(CLASS_NAMES):
            key = f"AP-{cls_name}"
            if key in bbox:
                metrics[f"val/AP50_{cls_name}"] = bbox[key]

        print(f"  mAP50-95: {bbox.get('AP', 0):.2f}")
        print(f"  mAP50: {bbox.get('AP50', 0):.2f}")

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
