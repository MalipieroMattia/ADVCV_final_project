#!/bin/bash
# Run 3 fine-tuning experiments for comparison in wandb

DATA_PATH="${1:-/work/Data_YOLO_unsplit}"

echo "=============================================="
echo "Experiment 1/3: Full Fine-tuning (freeze=0)"
echo "=============================================="
python main.py --mode train --config configs/config.yaml --data-path "$DATA_PATH" --freeze 0 --name full_finetune

echo "=============================================="
echo "Experiment 2/3: Partial Freeze (freeze=5)"
echo "=============================================="
python main.py --mode train --config configs/config.yaml --data-path "$DATA_PATH" --freeze 5 --name partial_freeze

echo "=============================================="
echo "Experiment 3/3: Freeze Backbone (freeze=10)"
echo "=============================================="
python main.py --mode train --config configs/config.yaml --data-path "$DATA_PATH" --freeze 10 --name freeze_backbone

echo "=============================================="
echo "All experiments complete! Check wandb for results."
echo "=============================================="

