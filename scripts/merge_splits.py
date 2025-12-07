#!/usr/bin/env python3
"""
Merge Split Data
================
Merge train/val/test folders back into a single unsplit dataset.

Usage:
    python scripts/merge_splits.py
    python scripts/merge_splits.py --input raw_data/Data_YOLO --output raw_data/Data_YOLO_unsplit
"""

import os
import shutil
import argparse
from pathlib import Path


def merge_splits(input_dir: Path, output_dir: Path) -> dict:
    """
    Merge train/val/test folders into single images/ and labels/ folders.
    
    Args:
        input_dir: Directory with images/{train,val,test} structure
        output_dir: Directory for merged data
        
    Returns:
        Dictionary with counts
    """
    # Create output directories
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    counts = {"images": 0, "labels": 0}
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    
    # Process each split
    for split in ["train", "val", "test"]:
        img_dir = input_dir / "images" / split
        lbl_dir = input_dir / "labels" / split
        
        if not img_dir.exists():
            continue
            
        print(f"Processing {split}...")
        
        # Copy images
        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() in extensions:
                dest = output_dir / "images" / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
                    counts["images"] += 1
                    
                    # Copy corresponding label
                    lbl_file = lbl_dir / (img_file.stem + ".txt")
                    if lbl_file.exists():
                        shutil.copy2(lbl_file, output_dir / "labels" / lbl_file.name)
                        counts["labels"] += 1
    
    return counts


def main():
    parser = argparse.ArgumentParser(description="Merge split YOLO data into single folder")
    parser.add_argument("--input", "-i", type=str, default="raw_data/Data_YOLO",
                        help="Input directory with split data")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: input_unsplit)")
    parser.add_argument("--in-place", action="store_true",
                        help="Merge in place (reorganize input directory)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    
    if args.in_place:
        # Create temp merged, then replace
        output_dir = input_dir.parent / (input_dir.name + "_merged_temp")
    elif args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_dir.parent / (input_dir.name + "_unsplit")
    
    print(f"\nMerging splits:")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}\n")
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    counts = merge_splits(input_dir, output_dir)
    
    print(f"\nMerged: {counts['images']} images, {counts['labels']} labels")
    
    if args.in_place:
        # Remove old split structure and rename
        print("\nReorganizing in place...")
        for split in ["train", "val", "test"]:
            for subdir in ["images", "labels"]:
                d = input_dir / subdir / split
                if d.exists():
                    shutil.rmtree(d)
        
        # Move merged files to original location
        for subdir in ["images", "labels"]:
            src = output_dir / subdir
            dst = input_dir / subdir
            for f in src.iterdir():
                shutil.move(str(f), str(dst / f.name))
        
        shutil.rmtree(output_dir)
        print(f"Reorganized {input_dir}")
    else:
        print(f"\nOutput saved to: {output_dir}")
    
    print("\nStructure for upload:")
    print("  Data_YOLO_unsplit/")
    print("  ├── images/")
    print("  │   ├── image1.jpg")
    print("  │   └── ...")
    print("  └── labels/")
    print("      ├── image1.txt")
    print("      └── ...")


if __name__ == "__main__":
    main()

