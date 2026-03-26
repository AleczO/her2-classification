import os
from pathlib import Path

# Current file location (src/config.py)
current_file = Path(__file__).resolve()

# Project root (her2-classification/)
BASE_DIR = current_file.parent.parent

# Check if running on Google Colab
IS_COLAB = Path('/content').exists()

if IS_COLAB:
    DATASET_ROOT = Path('/content/datasets/Patch-based-dataset')
    print("Running on Google Colab. Using /content/datasets as data root.")
else:
    DATASET_ROOT = BASE_DIR / "datasets" / "Patch-based-dataset"
    print(f"Running locally. Using {DATASET_ROOT} as data root.")

# Data Directories
TRAIN_DIR = DATASET_ROOT / "train_data_patch"
TEST_DIR = DATASET_ROOT / "test_data_patch"

# Results persistent storage
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"

# Ensure directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)