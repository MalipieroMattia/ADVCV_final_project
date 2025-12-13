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
COCO_IN = Path("raw_data/Data_COCO")
COCO_OUT = Path("raw_data/raw_data_unsplit/Data_COCO_unsplit")


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


def merge_coco():
    if (COCO_OUT / "annotations.json").exists():
        print("COCO: already done")
        return
    (COCO_OUT / "images").mkdir(parents=True, exist_ok=True)

    for split in ["train2017", "val2017"]:
        for f in (COCO_IN / split).glob("*.[jp][pn]g"):
            shutil.copy2(f, COCO_OUT / "images" / f.name)

    merged = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }
    for name in ["instances_train2017.json", "instances_val2017.json"]:
        with open(COCO_IN / "annotations" / name) as f:
            data = json.load(f)
        merged["images"].extend(data["images"])
        merged["annotations"].extend(data["annotations"])
        merged["categories"] = merged["categories"] or data["categories"]
        merged["info"] = merged["info"] or data.get(
            "info", {"description": "PCB Dataset"}
        )
        merged["licenses"] = merged["licenses"] or data.get("licenses", [])

    # Re-index annotation IDs to ensure uniqueness
    for i, ann in enumerate(merged["annotations"], start=1):
        ann["id"] = i

    with open(COCO_OUT / "annotations.json", "w") as f:
        json.dump(merged, f)
    print(
        f"COCO: merged {len(merged['images'])} images, {len(merged['annotations'])} annotations"
    )


if __name__ == "__main__":
    merge_yolo()
    merge_coco()
    print("Done! Upload raw_data/raw_data_unsplit/ to UCloud")
