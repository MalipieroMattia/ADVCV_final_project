"""
Evaluation Utilities - Lightweight wrapper around YOLO evaluation.
Only adds what YOLO doesn't provide: error analysis for EDA.
"""

import cv2 as cv
from pathlib import Path
from typing import Dict, Any, Optional, List
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

        # Extract metrics for return value
        metrics = {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }

        # Add per-class AP50
        if hasattr(results.box, "ap_class_index"):
            for i, cls_idx in enumerate(results.box.ap_class_index):
                cls_name = self.class_names.get(int(cls_idx), f"class_{cls_idx}")
                metrics[f"AP50_{cls_name}"] = float(results.box.ap50[i])

        print(f"\nEvaluation Results ({split}):")
        print(f"  mAP50: {metrics['mAP50']:.4f}")
        print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")

        # Run error analysis if requested
        if analyze_errors:
            errors = self._analyze_errors(data_yaml, split)
            self._log_errors_to_wandb(errors, prefix=split)

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

    def _log_errors_to_wandb(self, errors: List[Dict], prefix: str) -> bool:
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
                    round(e["confidence"], 4) if e["confidence"] else None,
                    round(e["iou"], 4),
                )
            wandb.log({f"{prefix}/error_analysis": table})

            # Log error counts
            counts = {}
            for e in errors:
                counts[e["error_type"]] = counts.get(e["error_type"], 0) + 1
            wandb.log({f"{prefix}/errors/{k}": v for k, v in counts.items()})
            wandb.log({f"{prefix}/errors/total": len(errors)})

            print(f"Logged {len(errors)} errors to wandb")
            return True

        except Exception as e:
            print(f"Could not log errors to wandb: {e}")
            return False
