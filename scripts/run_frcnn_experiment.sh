#!/bin/bash
# =============================================================================
# Run Faster R-CNN Training on UCloud
# =============================================================================
# 
# Supports TWO data formats:
#
# 1. UNSPLIT (recommended - matches YOLO workflow):
#    /work/Data_COCO_unsplit/
#    ├── images/
#    │   └── *.jpg
#    └── annotations.json
#
# 2. PRE-SPLIT (COCO 2017 format):
#    /work/Data_COCO/
#    ├── annotations/
#    │   ├── instances_train2017.json
#    │   └── instances_val2017.json
#    ├── train2017/
#    └── val2017/
#
# Usage:
#   bash scripts/run_frcnn_experiment.sh /work/Data_COCO_unsplit
#   bash scripts/run_frcnn_experiment.sh /work/Data_COCO_unsplit 100 4 my_run
#
# Prerequisites:
#   pip install 'git+https://github.com/facebookresearch/detectron2.git'
#   wandb login
# =============================================================================

DATA_PATH="${1:-/work/Data_COCO_unsplit}"
EPOCHS="${2:-100}"
BATCH="${3:-16}"  # 16 for H100, reduce if OOM
NAME="${4:-frcnn_train}"
MAX_TIME="${5:-90}"  # 90 minutes to match YOLO training time

echo "=============================================="
echo "Faster R-CNN Training (Detectron2)"
echo "=============================================="
echo "Data path: $DATA_PATH"
echo "Epochs: $EPOCHS (or $MAX_TIME min, whichever first)"
echo "Batch size: $BATCH"
echo "Run name: $NAME"
echo "=============================================="

python utils/train_frcnn.py \
    --data-path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --name "$NAME" \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42 \
    --max-time "$MAX_TIME"

echo "Training complete! Check WandB for results."

