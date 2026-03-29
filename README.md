# HER2 Breast Cancer Classification
### High-precision Deep Learning for Automated Histopathology Scoring

This repository implements a robust Deep Learning system for classifying HER2 status (0, 1+, 2+, 3+) in breast cancer histopathology patches using a fine-tuned ResNet-50 architecture.

## Performance Highlights
* **Accuracy**: **96.64%** (Fine-tuned on Phase 2)
* **Clinical Safety**: **99.56% Recall for Class 3+** (Minimizing False Negatives)
* **Interpretability**: Integrated **Grad-CAM** and **Early Feature Mapping** for decision verification.
* **Stability**: Optimized convergence using a hybrid SGD/Adam strategy with Learning Rate scheduling.

## Project Overview
Automated HER2 scoring is critical for determining targeted therapy eligibility in breast cancer patients. This project utilizes a **patch-based approach** to preserve cellular-level morphological details (membrane completeness and intensity), avoiding the information loss associated with resizing large Whole Slide Images (WSI).

## Technical Key Features
- **Explainable AI (XAI)**: Implementation of **Grad-CAM** to visualize "attention" regions and **feature map extraction** to verify early-layer pattern detection (edges, stain intensity).
- **Hybrid Configuration**: Dynamic `config.py` environment detection for seamless switching between local development (Windows) and high-performance cloud training (Google Colab/Linux).
- **Model Reliability**: Advanced calibration analysis using **Reliability Diagrams** and **Expected Calibration Error (ECE)** assessment.
- **Production-Ready Structure**: Core logic encapsulated in an installable `src/` package for modularity and reproducibility.

## Directory Structure
- `src/`: Modular library (Data Loaders, Model Architectures, Training Pipelines).
- `notebooks/`: Research, Inference, and Interpretability (Grad-CAM) notebooks.
- `results/`: Saved checkpoints (.pth), training logs, and evaluation plots.
- `datasets/`: Local data storage (standardized path structure).

## Project Roadmap
1. [x] Data exploration and patch preprocessing.
2. [x] Local CPU training smoke-test.
3. [x] Hybrid environment and repository setup.
4. [x] Full-scale GPU training (Phase 1: Baseline & Phase 2: Fine-tuning).
5. [x] Global Performance Evaluation (F1-Score, Confusion Matrix).
6. [x] Advanced Interpretability & Diagnostic Panel (Grad-CAM & Reliability Analysis).

## Installation & Usage
To set up the environment and install the project as an editable package:

```bash
# Clone the repository
git clone https://github.com/AleczO/her2-classification.git
cd her2-classification

# Install dependencies and the src package
pip install -r requirements.txt
pip install -e .
```

For detailed experimental results, including loss curves and calibration plots, please refer to [EXPERIMENTS.md](EXPERIMENTS.md).


##  Acknowledgements & Data Source

If you use this project or find the analysis helpful, please cite the original dataset providers:

> Md Serajun Nabi i in. “HER2-IHC-40x: A high-resolution histopathology dataset for HER2 IHC scoring in breast cancer”. W: Data in Brief 62 (2025), s. 111922. ISSN: 2352-3409.
> DOI: https://doi.org/10.1016/j.dib.2025.111922. 
> URL: https://www.sciencedirect.com/science/article/pii/S2352340925006468.


*The dataset consists of high-resolution (40x) histopathology patches, providing a critical benchmark for automated HER2 scoring systems.*