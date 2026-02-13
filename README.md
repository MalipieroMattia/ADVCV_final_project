# PCB Defect Detection (YOLOv8 Fine-Tuning)

This repository contains the code and analysis material for a PCB surface-defect detection study based on YOLOv8.  
The project compares fine-tuning strategies for `YOLOv8m` and `YOLOv8m-P2`, evaluates detection quality on held-out data, and includes post-training confidence-threshold analysis for operating-point selection.

## Scope of the Experiments

The study focuses on four model/strategy variants:
- `YOLOv8m-P2` (full fine-tuning)
- `YOLOv8m` (full fine-tuning)
- `YOLOv8m` (backbone frozen; neck + head trainable)
- `YOLOv8m` (head-only fine-tuning)

## Repository Layout

- `main.py`: main pipeline entry point (training, evaluation, prediction)
- `configs/`: experiment configurations
- `scripts/`: helper scripts for experiment execution
- `utils/`: data handling, training utilities, evaluation, logging, plots
- `EDA/`: dataset exploratory analysis notebook
- `analyses/`: result analysis, architecture inspection, and visualization notebooks
- `artifacts/`: locally stored run outputs used for offline analysis
