"""
Dataset Management Utilities
============================
Handles dataset loading, splitting, and YOLO data.yaml generation.
Supports both pre-split data and runtime stratified splitting.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
import yaml
import random


# PCB Defect Classes
CLASS_NAMES = {
    0: "SH",  # Short
    1: "SP",  # Spur
    2: "SC",  # Spurious Copper
    3: "OP",  # Open
    4: "MB",  # Mouse Bite
    5: "HB",  # Hole Breakout
    6: "CS",  # Conductor Scratch
    7: "CFO",  # Conductor Foreign Object
    8: "BMFO",  # Base Material Foreign Object
}


class DatasetManager:
    """Manage dataset paths and create YOLO data.yaml configurations."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dataset manager.

        Args:
            config: Configuration dictionary with data settings
        """
        self.config = config
        self.data_config = config.get("data", {})

        # Get data root (can be overridden by environment variable)
        self.data_root = os.environ.get("UCLOUD_DATA_PATH") or self.data_config.get(
            "root", "raw_data/Data_YOLO"
        )
        self.data_root = Path(self.data_root)

        # Check if data is pre-split or needs runtime splitting
        self.is_presplit = self._check_if_presplit()

    def _check_if_presplit(self) -> bool:
        """Check if dataset has train/val subdirectories."""
        train_dir = self.data_root / "images" / "train"
        return train_dir.exists()

    def create_data_yaml(
        self,
        output_path: str = "configs/data.yaml",
        class_names: Optional[Dict[int, str]] = None,
    ) -> str:
        """
        Create YOLO data.yaml configuration file.
        If data is not pre-split, performs stratified split first.

        Args:
            output_path: Where to save the data.yaml
            class_names: Optional custom class names dict

        Returns:
            Path to created data.yaml file
        """
        if class_names is None:
            class_names = self.data_config.get("class_names", CLASS_NAMES)

        # If not pre-split, do stratified split
        if not self.is_presplit:
            print("Data is not pre-split. Performing stratified split...")
            split_config = self.data_config.get("split", {})
            split_dir = self._stratified_split(
                val_ratio=split_config.get("val_ratio", 0.2),
                test_ratio=split_config.get("test_ratio", 0.0),
                seed=self.config.get("seed", 42),
            )
            data_path = split_dir
        else:
            data_path = self.data_root.resolve()

        # Build data config
        data_yaml = {
            "path": str(data_path),
            "train": self.data_config.get("train_images", "images/train"),
            "val": self.data_config.get("val_images", "images/val"),
            "nc": len(class_names),
            "names": class_names,
        }

        # Add test if it exists (from stratified split or pre-split)
        test_ratio = self.data_config.get("split", {}).get("test_ratio", 0)
        if test_ratio > 0 or (data_path / "images" / "test").exists():
            data_yaml["test"] = "images/test"

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        print(f"Created data.yaml at {output_path}")
        print(f"  Dataset path: {data_path}")
        print(f"  Classes: {len(class_names)}")

        return str(output_path)

    def _stratified_split(
        self,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        seed: int = 42,
    ) -> Path:
        """
        Perform stratified split preserving class balance.

        Args:
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            seed: Random seed

        Returns:
            Path to split dataset directory
        """
        random.seed(seed)

        images_dir = self.data_root / "images"
        labels_dir = self.data_root / "labels"

        # Get all image-label pairs grouped by their classes
        class_to_images = defaultdict(list)

        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                lbl_file = labels_dir / (img_file.stem + ".txt")
                if lbl_file.exists():
                    # Get classes in this image
                    classes = set()
                    with open(lbl_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                classes.add(int(parts[0]))

                    # Use primary class (most frequent or first) for stratification
                    # For multi-label, use a tuple of sorted classes as key
                    class_key = tuple(sorted(classes)) if classes else (-1,)
                    class_to_images[class_key].append((img_file, lbl_file))

        # Split each class group
        train_pairs = []
        val_pairs = []
        test_pairs = []

        for class_key, pairs in class_to_images.items():
            random.shuffle(pairs)
            n = len(pairs)
            n_test = int(n * test_ratio)
            n_val = int(n * val_ratio)

            test_pairs.extend(pairs[:n_test])
            val_pairs.extend(pairs[n_test : n_test + n_val])
            train_pairs.extend(pairs[n_test + n_val :])

        # Shuffle final sets
        random.shuffle(train_pairs)
        random.shuffle(val_pairs)
        random.shuffle(test_pairs)

        # Create output directory
        output_dir = self.data_root.parent / "PCB_Data"

        # Copy files to split directories
        for split_name, pairs in [
            ("train", train_pairs),
            ("val", val_pairs),
            ("test", test_pairs),
        ]:
            if not pairs:
                continue

            img_out = output_dir / "images" / split_name
            lbl_out = output_dir / "labels" / split_name
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            for img_path, lbl_path in pairs:
                shutil.copy2(img_path, img_out / img_path.name)
                shutil.copy2(lbl_path, lbl_out / lbl_path.name)

        print(f"\nStratified split (seed={seed}):")
        print(f"  Train: {len(train_pairs)} images")
        print(f"  Val:   {len(val_pairs)} images")
        if test_pairs:
            print(f"  Test:  {len(test_pairs)} images")
        print(f"  Output: {output_dir}")

        # Print class distribution
        self._print_split_distribution(train_pairs, val_pairs, test_pairs)

        return output_dir.resolve()

    def _print_split_distribution(
        self, train_pairs: List, val_pairs: List, test_pairs: List
    ) -> None:
        """Print class distribution across splits."""

        def count_classes(pairs):
            counts = defaultdict(int)
            for _, lbl_path in pairs:
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            counts[int(parts[0])] += 1
            return counts

        train_counts = count_classes(train_pairs)
        val_counts = count_classes(val_pairs)

        print("\n  Class distribution:")
        print(f"  {'Class':<6} {'Train':>8} {'Val':>8} {'Ratio':>8}")
        print(f"  {'-' * 32}")

        all_classes = sorted(set(train_counts.keys()) | set(val_counts.keys()))
        for cls_id in all_classes:
            train_n = train_counts.get(cls_id, 0)
            val_n = val_counts.get(cls_id, 0)
            total = train_n + val_n
            ratio = val_n / total if total > 0 else 0
            cls_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
            print(f"  {cls_name:<6} {train_n:>8} {val_n:>8} {ratio:>7.1%}")

    def verify_dataset(self) -> Tuple[bool, Dict[str, int]]:
        """
        Verify dataset exists and has expected structure.

        Returns:
            Tuple of (is_valid, counts_dict)
        """
        counts = {}
        is_valid = True

        print("\nVerifying dataset structure:")

        if self.is_presplit:
            # Check split structure
            required_dirs = [
                ("train_images", self.data_root / "images" / "train"),
                ("train_labels", self.data_root / "labels" / "train"),
                ("val_images", self.data_root / "images" / "val"),
                ("val_labels", self.data_root / "labels" / "val"),
            ]
        else:
            # Check unsplit structure
            required_dirs = [
                ("images", self.data_root / "images"),
                ("labels", self.data_root / "labels"),
            ]

        for name, dir_path in required_dirs:
            if dir_path.exists():
                count = len([f for f in dir_path.iterdir() if f.is_file()])
                counts[name] = count
                print(f"  ✓ {name}: {count} files")
            else:
                counts[name] = 0
                is_valid = False
                print(f"  ✗ {name}: NOT FOUND")

        if not self.is_presplit:
            print(f"\n  Mode: Unsplit data (will split at training time)")
        else:
            print(f"\n  Mode: Pre-split data")

        return is_valid, counts

    def get_class_distribution(self) -> Dict[int, int]:
        """
        Count annotations per class in the dataset.

        Returns:
            Dictionary mapping class_id to count
        """
        distribution = {i: 0 for i in range(len(CLASS_NAMES))}

        # Find labels directory
        if self.is_presplit:
            labels_dir = self.data_root / "labels" / "train"
        else:
            labels_dir = self.data_root / "labels"

        if not labels_dir.exists():
            return distribution

        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if class_id in distribution:
                            distribution[class_id] += 1

        return distribution


def unsplit_dataset(split_data_dir: str, output_dir: str) -> None:
    """
    Merge train/val split folders back into single images and labels folders.

    Args:
        split_data_dir: Path to directory containing images/{train,val} and labels/{train,val}
        output_dir: Path where merged images/ and labels/ folders will be created
    """
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    for subset in ["train", "val", "test"]:
        images_subset_dir = f"{split_data_dir}/images/{subset}"
        labels_subset_dir = f"{split_data_dir}/labels/{subset}"

        if not os.path.exists(images_subset_dir):
            continue

        for image in os.listdir(images_subset_dir):
            if image.endswith((".jpg", ".jpeg", ".png")):
                shutil.copy(
                    f"{images_subset_dir}/{image}", f"{output_dir}/images/{image}"
                )

                label_file = os.path.splitext(image)[0] + ".txt"
                label_path = f"{labels_subset_dir}/{label_file}"
                if os.path.exists(label_path):
                    shutil.copy(label_path, f"{output_dir}/labels/{label_file}")

    print(f"Unsplit complete! Merged data saved to {output_dir}")


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    is_already_split: bool = False,
    random_state: int = 42,
) -> Dict[str, int]:
    """
    Split dataset into train/val/test sets.

    Note: For stratified splitting, use DatasetManager instead.
    """
    from sklearn.model_selection import train_test_split

    if is_already_split:
        temp_unsplit_dir = f"{output_dir}_temp_unsplit"
        unsplit_dataset(images_dir, temp_unsplit_dir)
        images_dir = f"{temp_unsplit_dir}/images"
        labels_dir = f"{temp_unsplit_dir}/labels"

    images = [
        f for f in os.listdir(images_dir) if f.endswith((".jpg", ".jpeg", ".png"))
    ]

    train_val_images, test_images = train_test_split(
        images, test_size=test_size, random_state=random_state
    )
    train_images, val_images = train_test_split(
        train_val_images, test_size=val_size, random_state=random_state
    )

    for subset, subset_images in [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images),
    ]:
        os.makedirs(f"{output_dir}/images/{subset}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{subset}", exist_ok=True)

        for image in subset_images:
            shutil.copy(
                f"{images_dir}/{image}", f"{output_dir}/images/{subset}/{image}"
            )
            label_file = os.path.splitext(image)[0] + ".txt"
            src_label = f"{labels_dir}/{label_file}"
            if os.path.exists(src_label):
                shutil.copy(src_label, f"{output_dir}/labels/{subset}/{label_file}")

    if is_already_split:
        temp_unsplit_dir = f"{output_dir}_temp_unsplit"
        if os.path.exists(temp_unsplit_dir):
            shutil.rmtree(temp_unsplit_dir)

    counts = {
        "train": len(train_images),
        "val": len(val_images),
        "test": len(test_images),
    }

    print(f"Dataset split saved to {output_dir}")
    print(f"  Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']}")

    return counts
