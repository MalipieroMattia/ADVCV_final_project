"""
Evaluation and Inference Utilities
===================================
Handles model evaluation, inference, and result visualization.
"""

import cv2 as cv
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

from ultralytics import YOLO

from utils.data_loader import CLASS_NAMES


class YOLOEvaluator:
    """Evaluate YOLO models and run inference."""

    def __init__(
        self, 
        model_path: str,
        class_names: Optional[Dict[int, str]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained YOLO weights
            class_names: Optional custom class names
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.class_names = class_names or CLASS_NAMES
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def evaluate(
        self, 
        data_yaml: str,
        split: str = "val",
        save_results: bool = True,
        project: str = "runs/evaluate",
        name: str = "eval",
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset.

        Args:
            data_yaml: Path to data.yaml file
            split: Dataset split to evaluate ('val' or 'test')
            save_results: Whether to save evaluation results
            project: Project directory for results
            name: Name for this evaluation run

        Returns:
            Dictionary of evaluation metrics
        """
        results = self.model.val(
            data=data_yaml,
            split=split,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=save_results,
            project=project,
            name=name,
            plots=True,
        )
        
        # Extract key metrics
        metrics = {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }
        
        # Per-class metrics
        if hasattr(results.box, "ap_class_index"):
            for i, cls_idx in enumerate(results.box.ap_class_index):
                cls_name = self.class_names.get(int(cls_idx), f"class_{cls_idx}")
                metrics[f"AP50_{cls_name}"] = float(results.box.ap50[i])
        
        self._print_metrics(metrics)
        
        return metrics

    def _print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print evaluation metrics."""
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 50 + "\n")

    def predict(
        self,
        source: Union[str, np.ndarray, List],
        save: bool = False,
        project: str = "runs/predict",
        name: str = "predict",
    ) -> List[Any]:
        """
        Run inference on images.

        Args:
            source: Image path, numpy array, or list of images
            save: Whether to save predictions
            project: Project directory for results
            name: Name for this prediction run

        Returns:
            List of prediction results
        """
        results = self.model.predict(
            source=source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=save,
            project=project,
            name=name,
        )
        
        return results

    def predict_and_visualize(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Run prediction and visualize results.

        Args:
            image_path: Path to input image
            output_path: Optional path to save visualization
            show: Whether to display the image

        Returns:
            Tuple of (annotated_image, detections_list)
        """
        # Read image
        image = cv.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run prediction
        results = self.predict(image_path)[0]
        
        # Extract detections
        detections = []
        for box in results.boxes:
            det = {
                "class_id": int(box.cls),
                "class_name": self.class_names.get(int(box.cls), "unknown"),
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist(),
            }
            detections.append(det)
        
        # Draw boxes on image
        annotated = self._draw_boxes(image, detections)
        
        # Save if requested
        if output_path:
            cv.imwrite(output_path, annotated)
            print(f"Saved visualization to {output_path}")
        
        # Show if requested
        if show:
            cv.imshow("Predictions", annotated)
            cv.waitKey(0)
            cv.destroyAllWindows()
        
        return annotated, detections

    def _draw_boxes(
        self, 
        image: np.ndarray, 
        detections: List[Dict],
        colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """Draw bounding boxes on image."""
        if colors is None:
            np.random.seed(42)
            colors = {
                i: tuple(map(int, np.random.randint(0, 255, 3)))
                for i in range(len(self.class_names))
            }
        
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cls_id = det["class_id"]
            conf = det["confidence"]
            cls_name = det["class_name"]
            
            color = colors.get(cls_id, (0, 255, 0))
            
            cv.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"{cls_name}: {conf:.2f}"
            (label_w, label_h), baseline = cv.getTextSize(
                label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv.rectangle(
                annotated,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1,
            )
            cv.putText(
                annotated,
                label,
                (x1, y1 - baseline - 2),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        
        return annotated

    def batch_predict(
        self,
        image_dir: str,
        output_dir: Optional[str] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> List[Dict]:
        """Run predictions on all images in a directory."""
        image_dir = Path(image_dir)
        image_files = [
            f for f in image_dir.iterdir()
            if f.suffix.lower() in extensions
        ]
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        for img_path in image_files:
            output_path = None
            if output_dir:
                output_path = str(output_dir / img_path.name)
            
            try:
                _, detections = self.predict_and_visualize(
                    str(img_path),
                    output_path=output_path,
                )
                all_results.append({
                    "image": str(img_path),
                    "detections": detections,
                    "num_defects": len(detections),
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                all_results.append({
                    "image": str(img_path),
                    "error": str(e),
                })
        
        total_detections = sum(
            r.get("num_defects", 0) for r in all_results if "error" not in r
        )
        print(f"\nProcessed {len(all_results)} images")
        print(f"Total defects detected: {total_detections}")
        
        return all_results

    def export_model(
        self,
        format: str = "onnx",
        output_dir: str = "exports",
        **kwargs,
    ) -> str:
        """Export model to different format."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        export_path = self.model.export(format=format, **kwargs)
        print(f"Exported model to {export_path}")
        return export_path

