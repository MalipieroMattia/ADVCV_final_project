#!/usr/bin/env python3
"""
Reshuffle YOLO Dataset
======================
Re-split a YOLO format dataset with different ratios or random seeds.

Usage:
    python scripts/reshuffle_data.py
    python scripts/reshuffle_data.py --seed 123
    python scripts/reshuffle_data.py --val 0.15 --test 0.15
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict


def get_image_label_pairs(images_dir: Path, labels_dir: Path) -> list:
    """Find all image-label pairs."""
    pairs = []
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for img_file in images_dir.iterdir():
        if img_file.suffix.lower() in extensions:
            label_file = labels_dir / (img_file.stem + '.txt')
            if label_file.exists():
                pairs.append((img_file, label_file))
    
    return pairs


def collect_all_data(input_dir: Path) -> list:
    """Collect all image-label pairs from subdirectories."""
    all_pairs = []
    images_root = input_dir / "images"
    labels_root = input_dir / "labels"
    
    for split in ["train", "val", "test"]:
        img_dir = images_root / split
        lbl_dir = labels_root / split
        
        if img_dir.exists() and lbl_dir.exists():
            pairs = get_image_label_pairs(img_dir, lbl_dir)
            all_pairs.extend(pairs)
            print(f"  Found {len(pairs)} pairs in {split}/")
    
    return all_pairs


def split_data(pairs: list, val_ratio: float, test_ratio: float, seed: int) -> dict:
    """Split data into train/val/test sets."""
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val
    
    return {
        'train': shuffled[:n_train],
        'val': shuffled[n_train:n_train + n_val],
        'test': shuffled[n_train + n_val:] if test_ratio > 0 else [],
    }


def copy_pairs_to_split(pairs: list, output_dir: Path, split_name: str) -> None:
    """Copy pairs to output split directory."""
    img_out = output_dir / "images" / split_name
    lbl_out = output_dir / "labels" / split_name
    
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    
    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, img_out / img_path.name)
        shutil.copy2(lbl_path, lbl_out / lbl_path.name)


def main():
    parser = argparse.ArgumentParser(description="Reshuffle YOLO dataset")
    parser.add_argument("--input", "-i", type=str, default="raw_data/Data_YOLO")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir
    
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Split: train={1-args.val-args.test:.0%}, val={args.val:.0%}, test={args.test:.0%}")
    print(f"Seed: {args.seed}\n")
    
    print("Collecting data...")
    all_pairs = collect_all_data(input_dir)
    
    if not all_pairs:
        print("Error: No image-label pairs found!")
        return
    
    print(f"\nTotal pairs: {len(all_pairs)}")
    
    print(f"\nShuffling with seed={args.seed}...")
    splits = split_data(all_pairs, args.val, args.test, args.seed)
    
    print(f"\nNew split:")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val:   {len(splits['val'])} images")
    if args.test > 0:
        print(f"  Test:  {len(splits['test'])} images")
    
    if args.dry_run:
        print("\n[DRY RUN] No files copied.")
        return
    
    if output_dir == input_dir:
        print("\nClearing existing directories...")
        for split in ["train", "val", "test"]:
            for subdir in ["images", "labels"]:
                d = output_dir / subdir / split
                if d.exists():
                    shutil.rmtree(d)
    
    print("\nCopying files...")
    for split_name, pairs in splits.items():
        if pairs:
            copy_pairs_to_split(pairs, output_dir, split_name)
            print(f"  {split_name}: {len(pairs)} pairs copied")
    
    print(f"\nDone! Output: {output_dir}")


if __name__ == "__main__":
    main()

