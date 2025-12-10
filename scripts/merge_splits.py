#!/usr/bin/env python3
"""
Merge split datasets into unsplit format for UCloud.
Usage: python scripts/merge_splits.py
"""

import json
import shutil
from pathlib import Path

# Paths
YOLO_IN = Path("raw_data/Data_YOLO")
YOLO_OUT = Path("raw_data/raw_data_unsplit/Data_YOLO_unsplit")


def merge_yolo():
    if (YOLO_OUT / "images").exists():
        print("YOLO: already done")
        return
    (YOLO_OUT / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_OUT / "labels").mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        img_dir = YOLO_IN / "images" / split
        if not img_dir.exists():
            continue
        for f in img_dir.glob("*.[jp][pn]g"):
            shutil.copy2(f, YOLO_OUT / "images" / f.name)
            lbl = YOLO_IN / "labels" / split / (f.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, YOLO_OUT / "labels" / lbl.name)
    print(f"YOLO: merged to {YOLO_OUT}")


if __name__ == "__main__":
    merge_yolo()
    print("Done! Upload raw_data/raw_data_unsplit/ to UCloud")
