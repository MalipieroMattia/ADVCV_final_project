#!/bin/bash
# Run 3 fine-tuning experiments + YOLOv8m-p2 full fine-tune

DATA_PATH="${1:-/work/Data_YOLO_unsplit}"

echo "=============================================="
echo "Experiment 1/4: YOLOv8m-p2 Full Fine-tuning (freeze=0)"
echo "=============================================="
python main.py --mode train --config configs/config_targeting_small_objects.yaml --data-path "$DATA_PATH" --freeze 0 --name yolov8m_p2_full

echo "=============================================="
echo "Experiment 2/4: Full Fine-tuning (freeze=0)"
echo "=============================================="
python main.py --mode train --config configs/config.yaml --data-path "$DATA_PATH" --freeze 0 --name full_finetune

echo "=============================================="
echo "Experiment 3/4: Freeze Backbone, Train Neck+Head (freeze=10)"
echo "=============================================="
python main.py --mode train --config configs/config.yaml --data-path "$DATA_PATH" --freeze 10 --name freeze_backbone

echo "=============================================="
echo "Experiment 4/4: Train Head Only (freeze=22)"
echo "=============================================="
python main.py --mode train --config configs/config.yaml --data-path "$DATA_PATH" --freeze 22 --name head_only

echo "=============================================="
echo "All experiments complete! Check wandb for results."
echo "=============================================="
