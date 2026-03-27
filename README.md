# HER2 Breast Cancer Classification

Deep Learning system for HER2 status classification (0, 1+, 2+, 3+) in breast cancer histopathology patches using ResNet-50.

## Project Overview
This project implements an automated pipeline for scoring HER2 from tissue patches. It utilizes a patch-based approach to preserve cellular-level morphological details, avoiding the information loss associated with resizing large Whole Slide Images (WSI).

## Technical Specifications
- **Hybrid Configuration**: Dynamic `config.py` environment detection for seamless switching between local (Windows) and Google Colab (Linux) runtimes.
- **Model Interpretability**: Implementation of first-layer feature map visualizations to analyze neural network pattern detection.
- **Architecture**: ResNet-50 with ImageNet pretrained weights, customized for 4-class medical image classification.
- **Modularity**: Core logic encapsulated in an installable `src/` package for reproducible experimentation.

## Directory Structure
- `src/`: Core library (Data Loaders, Models, Training, Utilities).
- `notebooks/`: Research and inference Jupyter notebooks.
- `results/`: Saved model checkpoints (.pth) and evaluation plots.
- `datasets/`: Local data storage (ignored by git).

## Project Plan
1. [x] Data exploration and patch preprocessing.
2. [x] Local CPU training smoke-test (Initial 3 epochs).
3. [x] Hybrid environment and repository setup.
4. [ ] Full-scale GPU training on Google Colab (Target: 50+ epochs).
5. [ ] Performance evaluation (F1-Score, Confusion Matrix).
6. [ ] Advanced interpretability (Grad-CAM).

## Installation
To install the project as an editable package:
```bash
pip install -e .
```