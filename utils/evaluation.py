"""
Evaluation Utilities - Lightweight wrapper around YOLO evaluation.
Only adds what YOLO doesn't provide: error analysis for EDA.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, List
import time

import cv2 as cv
import yaml

from ultralytics import YOLO
from utils.data_loader import CLASS_NAMES


class YOLOEvaluator:
    """Thin wrapper around YOLO evaluation with error analysis."""

    def __init__(
        self,
        model_path: str,
        class_names: Optional[Dict[int, str]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ):
        self.model = YOLO(model_path)
        self.class_names = class_names or CLASS_NAMES
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        data_yaml: str,
        split: str = "val",
        project: str = "runs/evaluate",
        name: str = "eval",
        analyze_errors: bool = True,
        compute_coco_metrics: bool = False,
    ) -> Dict[str, Any]:
        """
        Run YOLO evaluation and optionally analyze errors.

        YOLO handles wandb logging automatically when enabled.
        """
        # Run YOLO validation (handles wandb logging internally)
        results = self.model.val(
            data=data_yaml,
            split=split,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            project=project,
            name=name,
            plots=True,
        )
        # Use YOLO's actual output directory to avoid split files across
        # test_evaluation/ and test_evaluation2/ when name collisions happen.
        output_dir = Path(results.save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract metrics for return value
        metrics = {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }

        # Add per-class metrics if available
        if hasattr(results.box, "ap_class_index"):
            for i, cls_idx in enumerate(results.box.ap_class_index):
                cls_name = self.class_names.get(int(cls_idx), f"class_{cls_idx}")
                if hasattr(results.box, "ap50") and i < len(results.box.ap50):
                    metrics[f"AP50_{cls_name}"] = float(results.box.ap50[i])
                if hasattr(results.box, "ap") and i < len(results.box.ap):
                    metrics[f"AP50-95_{cls_name}"] = float(results.box.ap[i])
                if hasattr(results.box, "p") and i < len(results.box.p):
                    metrics[f"Precision_{cls_name}"] = float(results.box.p[i])
                if hasattr(results.box, "r") and i < len(results.box.r):
                    metrics[f"Recall_{cls_name}"] = float(results.box.r[i])

        print(f"\nEvaluation Results ({split}):")
        print(f"  mAP50: {metrics['mAP50']:.4f}")
        print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")

        # Run error analysis if requested
        if analyze_errors:
            errors = self._analyze_errors(data_yaml, split)
            self._save_errors_csv(errors, output_dir / "errors.csv")
            misclass_summary = self._summarize_misclassifications(errors)
            self._save_misclass_summary(
                misclass_summary, output_dir / "misclass_summary.csv"
            )
            self._log_errors_to_wandb(
                errors, prefix=split, misclass_summary=misclass_summary
            )

        if compute_coco_metrics:
            coco_metrics = self._compute_coco_size_metrics(data_yaml, split)
            if coco_metrics:
                metrics.update(coco_metrics)
                self._log_coco_metrics_to_wandb(coco_metrics, prefix=split)

        metrics_to_save = dict(metrics)
        metrics_to_save["split"] = split
        metrics_to_save["conf_threshold"] = self.conf_threshold
        metrics_to_save["iou_threshold"] = self.iou_threshold
        self._save_metrics_csv(metrics_to_save, output_dir / "metrics.csv")
        self._log_eval_files_artifact(output_dir, split)

        return metrics

    def _analyze_errors(self, data_yaml: str, split: str) -> List[Dict[str, Any]]:
        """Compare predictions to ground truth and collect errors."""
        with open(data_yaml, "r") as f:
            data_config = yaml.safe_load(f)

        base_path = Path(data_config["path"])
        images_path = base_path / data_config.get(split, f"images/{split}")
        labels_path = base_path / "labels" / split

        errors = []
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        print(f"\nAnalyzing errors on {len(image_files)} images...")

        for img_path in image_files:
            # Load ground truth
            label_path = labels_path / (img_path.stem + ".txt")
            gt_boxes = self._load_labels(label_path)

            # Get predictions
            results = self.model.predict(
                str(img_path), conf=self.conf_threshold, verbose=False
            )
            pred_boxes = self._extract_predictions(results)

            # Get image dimensions for coordinate conversion
            img = cv.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Convert GT from YOLO format to xyxy
            for gt in gt_boxes:
                cx, cy, bw, bh = gt["bbox_yolo"]
                gt["bbox_xyxy"] = [
                    (cx - bw / 2) * w,
                    (cy - bh / 2) * h,
                    (cx + bw / 2) * w,
                    (cy + bh / 2) * h,
                ]

            # Match and find errors
            errors.extend(
                self._match_and_find_errors(img_path.name, gt_boxes, pred_boxes)
            )

        # Print summary
        error_counts = {}
        for e in errors:
            error_counts[e["error_type"]] = error_counts.get(e["error_type"], 0) + 1
        print(f"Found {len(errors)} errors: {error_counts}")

        return errors

    def _load_labels(self, label_path: Path) -> List[Dict]:
        """Load YOLO format labels."""
        boxes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        boxes.append(
                            {
                                "class_id": cls_id,
                                "class_name": self.class_names.get(
                                    cls_id, f"class_{cls_id}"
                                ),
                                "bbox_yolo": [float(x) for x in parts[1:5]],
                            }
                        )
        return boxes

    def _extract_predictions(self, results) -> List[Dict]:
        """Extract predictions from YOLO results."""
        boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                boxes.append(
                    {
                        "class_id": int(box.cls),
                        "class_name": self.class_names.get(
                            int(box.cls), f"class_{int(box.cls)}"
                        ),
                        "bbox_xyxy": box.xyxy[0].tolist(),
                        "confidence": float(box.conf),
                    }
                )
        return boxes

    def _match_and_find_errors(
        self, image_name: str, gt_boxes: List[Dict], pred_boxes: List[Dict]
    ) -> List[Dict]:
        """Match predictions to GT and return errors."""
        errors = []
        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)

        # Match predictions to ground truth by IoU
        for pi, pred in enumerate(pred_boxes):
            best_iou, best_gt_idx = 0, -1
            for gi, gt in enumerate(gt_boxes):
                if gt_matched[gi]:
                    continue
                iou = self._compute_iou(pred["bbox_xyxy"], gt["bbox_xyxy"])
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, gi

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                gt = gt_boxes[best_gt_idx]
                gt_matched[best_gt_idx] = True
                pred_matched[pi] = True
                # Wrong class error
                if pred["class_id"] != gt["class_id"]:
                    errors.append(
                        {
                            "image": image_name,
                            "error_type": "wrong_class",
                            "true_class": gt["class_name"],
                            "pred_class": pred["class_name"],
                            "confidence": pred["confidence"],
                            "iou": best_iou,
                        }
                    )

        # False positives (unmatched predictions)
        for pi, pred in enumerate(pred_boxes):
            if not pred_matched[pi]:
                errors.append(
                    {
                        "image": image_name,
                        "error_type": "false_positive",
                        "true_class": None,
                        "pred_class": pred["class_name"],
                        "confidence": pred["confidence"],
                        "iou": 0,
                    }
                )

        # Missed detections (unmatched ground truth)
        for gi, gt in enumerate(gt_boxes):
            if not gt_matched[gi]:
                errors.append(
                    {
                        "image": image_name,
                        "error_type": "missed",
                        "true_class": gt["class_name"],
                        "pred_class": None,
                        "confidence": None,
                        "iou": 0,
                    }
                )

        return errors

    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two xyxy boxes."""
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def _log_errors_to_wandb(
        self,
        errors: List[Dict],
        prefix: str,
        misclass_summary: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Log error analysis table to wandb."""
        try:
            import wandb

            if wandb.run is None:
                print("No active wandb run, skipping error logging")
                return False

            if not errors:
                print("No errors to log")
                return True

            # Create wandb table
            table = wandb.Table(
                columns=[
                    "image",
                    "error_type",
                    "true_class",
                    "pred_class",
                    "confidence",
                    "iou",
                ]
            )
            for e in errors:
                table.add_data(
                    e["image"],
                    e["error_type"],
                    e["true_class"] or "",
                    e["pred_class"] or "",
                    round(e["confidence"], 4) if e["confidence"] is not None else None,
                    round(e["iou"], 4) if e["iou"] is not None else None,
                )
            wandb.log({f"{prefix}/error_analysis": table})

            # Log error counts
            counts = {}
            for e in errors:
                counts[e["error_type"]] = counts.get(e["error_type"], 0) + 1
            wandb.log({f"{prefix}/errors/{k}": v for k, v in counts.items()})
            wandb.log({f"{prefix}/errors/total": len(errors)})

            if misclass_summary:
                summary_table = wandb.Table(
                    columns=[
                        "true_class",
                        "pred_class",
                        "count",
                        "avg_confidence",
                        "avg_iou",
                    ]
                )
                for row in misclass_summary:
                    summary_table.add_data(
                        row.get("true_class", ""),
                        row.get("pred_class", ""),
                        row.get("count", 0),
                        row.get("avg_confidence", None),
                        row.get("avg_iou", None),
                    )
                wandb.log({f"{prefix}/misclass_summary": summary_table})

            print(f"Logged {len(errors)} errors to wandb")
            return True

        except Exception as e:
            print(f"Could not log errors to wandb: {e}")
            return False

    def _log_coco_metrics_to_wandb(self, coco_metrics: Dict[str, float], prefix: str) -> None:
        """Log COCO-style metrics to wandb with a split prefix."""
        try:
            import wandb

            if wandb.run is None:
                return

            wandb.log({f"{prefix}/{k}": v for k, v in coco_metrics.items()})
        except Exception as e:
            print(f"Could not log COCO metrics to wandb: {e}")

    def _log_eval_files_artifact(self, output_dir: Path, split: str) -> None:
        """Upload evaluation CSV files as a W&B artifact tied to the current run."""
        try:
            import wandb

            if wandb.run is None:
                print("No active wandb run, skipping eval artifact upload")
                return

            files = [
                output_dir / "metrics.csv",
                output_dir / "errors.csv",
                output_dir / "misclass_summary.csv",
            ]
            existing_files = [p for p in files if p.exists()]
            if not existing_files:
                print("No evaluation CSV files found to upload")
                return

            artifact = wandb.Artifact(
                name=f"eval_{split}_{wandb.run.id}_{int(time.time())}",
                type="evaluation",
                metadata={
                    "split": split,
                    "conf_threshold": self.conf_threshold,
                    "iou_threshold": self.iou_threshold,
                },
            )
            for p in existing_files:
                artifact.add_file(str(p), name=p.name)

            wandb.log_artifact(artifact, aliases=[split, "latest"])
            print(f"Uploaded evaluation artifact for {split} with {len(existing_files)} files")
        except Exception as e:
            print(f"Could not upload evaluation artifact: {e}")

    def _save_metrics_csv(self, metrics: Dict[str, Any], output_path: Path) -> None:
        """Save evaluation metrics to a CSV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, value in metrics.items():
                writer.writerow([key, value])

    def _save_errors_csv(self, errors: List[Dict], output_path: Path) -> None:
        """Save detailed error list to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["image", "error_type", "true_class", "pred_class", "confidence", "iou"]
            )
            for e in errors:
                writer.writerow(
                    [
                        e.get("image", ""),
                        e.get("error_type", ""),
                        e.get("true_class", "") or "",
                        e.get("pred_class", "") or "",
                        e.get("confidence", ""),
                        e.get("iou", ""),
                    ]
                )

    def _summarize_misclassifications(
        self, errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Summarize wrong-class errors into a confusion-style table."""
        summary = defaultdict(lambda: {"count": 0, "conf_sum": 0.0, "iou_sum": 0.0})

        for e in errors:
            if e.get("error_type") != "wrong_class":
                continue
            true_class = e.get("true_class", "")
            pred_class = e.get("pred_class", "")
            key = (true_class, pred_class)
            summary[key]["count"] += 1
            if e.get("confidence") is not None:
                summary[key]["conf_sum"] += float(e["confidence"])
            if e.get("iou") is not None:
                summary[key]["iou_sum"] += float(e["iou"])

        rows = []
        for (true_class, pred_class), stats in summary.items():
            count = stats["count"]
            rows.append(
                {
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "count": count,
                    "avg_confidence": stats["conf_sum"] / count if count else None,
                    "avg_iou": stats["iou_sum"] / count if count else None,
                }
            )

        rows.sort(key=lambda r: r["count"], reverse=True)
        return rows

    def _save_misclass_summary(
        self, summary_rows: List[Dict[str, Any]], output_path: Path
    ) -> None:
        """Save misclassification summary to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["true_class", "pred_class", "count", "avg_confidence", "avg_iou"]
            )
            for row in summary_rows:
                writer.writerow(
                    [
                        row.get("true_class", ""),
                        row.get("pred_class", ""),
                        row.get("count", 0),
                        row.get("avg_confidence", ""),
                        row.get("avg_iou", ""),
                    ]
                )

    def _compute_coco_size_metrics(self, data_yaml: str, split: str) -> Dict[str, float]:
        """
        Compute COCO-style AP/AR for small/medium/large objects using pycocotools.

        Size buckets (COCO):
        - small: area < 32^2
        - medium: 32^2 <= area < 96^2
        - large: area >= 96^2
        """
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except Exception as e:
            print(f"pycocotools not available, skipping COCO metrics: {e}")
            return {}

        with open(data_yaml, "r") as f:
            data_config = yaml.safe_load(f)

        base_path = Path(data_config["path"])
        images_path = base_path / data_config.get(split, f"images/{split}")
        labels_path = base_path / "labels" / split

        image_files = sorted(
            list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        )
        if not image_files:
            print("No images found for COCO evaluation")
            return {}

        # Build COCO ground truth
        images = []
        annotations = []
        categories = []
        for cls_id in sorted(self.class_names.keys()):
            categories.append({"id": cls_id + 1, "name": self.class_names[cls_id]})

        ann_id = 1
        image_id_map: Dict[str, int] = {}
        for idx, img_path in enumerate(image_files, start=1):
            img = cv.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            images.append(
                {"id": idx, "file_name": img_path.name, "width": w, "height": h}
            )
            image_id_map[img_path.name] = idx

            label_path = labels_path / (img_path.stem + ".txt")
            if label_path.exists():
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x = (cx - bw / 2) * w
                        y = (cy - bh / 2) * h
                        box_w = bw * w
                        box_h = bh * h
                        area = max(0.0, box_w) * max(0.0, box_h)
                        annotations.append(
                            {
                                "id": ann_id,
                                "image_id": idx,
                                "category_id": cls_id + 1,
                                "bbox": [x, y, box_w, box_h],
                                "area": area,
                                "iscrowd": 0,
                            }
                        )
                        ann_id += 1

        if not annotations:
            print("No annotations found for COCO evaluation")
            return {}

        coco_gt = COCO()
        coco_gt.dataset = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        coco_gt.createIndex()

        # Build detections
        detections = []
        for img_path in image_files:
            image_id = image_id_map.get(img_path.name)
            if image_id is None:
                continue

            results = self.model.predict(
                str(img_path),
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            if len(results) == 0 or results[0].boxes is None:
                continue

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_w = max(0.0, x2 - x1)
                box_h = max(0.0, y2 - y1)
                detections.append(
                    {
                        "image_id": image_id,
                        "category_id": int(box.cls) + 1,
                        "bbox": [x1, y1, box_w, box_h],
                        "score": float(box.conf),
                    }
                )

        if not detections:
            print("No detections found for COCO evaluation")
            return {}

        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        coco_metrics = {
            "coco/AP50-95": float(stats[0]),
            "coco/AP50": float(stats[1]),
            "coco/AP75": float(stats[2]),
            "coco/APs": float(stats[3]),
            "coco/APm": float(stats[4]),
            "coco/APl": float(stats[5]),
            "coco/AR": float(stats[8]),
            "coco/ARs": float(stats[9]),
            "coco/ARm": float(stats[10]),
            "coco/ARl": float(stats[11]),
        }

        return coco_metrics
