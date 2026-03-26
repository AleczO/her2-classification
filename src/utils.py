import torch
import matplotlib.pyplot as plt
from pathlib import Path
from .config import MODELS_DIR, PLOTS_DIR

def count_classes(data_path):
    path = Path(data_path)
    if not path.exists():
        return {}
    
    counts = {d.name: len(list(d.glob('*'))) for d in path.iterdir() if d.is_dir()}
    return dict(sorted(counts.items()))


def calculate_class_weights(counts_dict):
    total = sum(counts_dict.values())
    
    weights = {cls: total / count for cls, count in counts_dict.items()}
    return weights


def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Save model checkpoint to results/models/ directory.
    """

    path = MODELS_DIR / filename
    torch.save(state, path)
    print(f"Checkpoint saved successfully at: {path}")


def plot_history(history, filename="training_curves.png"):
    """Plots Loss and Accuracy."""
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    
    save_path = PLOTS_DIR / filename
    plt.savefig(save_path)
    plt.close()
    print(f"Wykresy zapisane w: {save_path}")