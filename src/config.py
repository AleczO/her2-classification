import os
from pathlib import Path

# Current file location (src/config.py)
current_file = Path(__file__).resolve()

# Project root (her2-classification/)
BASE_DIR = current_file.parent.parent

# Check if running on Google Colab environment
IS_COLAB = Path('/content').exists()

if IS_COLAB:
    # Colab data to local /content extraction
    DATASET_ROOT = Path('/content/datasets') 
    
    RESULTS_DIR = Path('/content/drive/MyDrive/HER2_Output')
    print(f"Environment: Google Colab. Saving results to Drive: {RESULTS_DIR}")
else:

    # Local environment setup
    DATASET_ROOT = BASE_DIR / "datasets" / "Patch-based-dataset"
    RESULTS_DIR = BASE_DIR / "results"
    print(f"Environment: Local Machine. Saving results to: {RESULTS_DIR}")

# Data Paths (Adjusted for patch-based dataset structure)
TRAIN_DIR = DATASET_ROOT / "train_data_patch"
TEST_DIR = DATASET_ROOT / "test_data_patch"

# Result Subdirectories
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"

# Ensure all result directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 7