import torch
import matplotlib.pyplot as plt
from pathlib import Path
from .config import MODELS_DIR, PLOTS_DIR
import seaborn as sns
import pandas as pd
import os

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



def plot_history(history, save_path=None):
    """
    Plots training history
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set_theme(style="whitegrid")
    
    epochs_count = len(history['train_acc'])
    epochs = list(range(1, epochs_count + 1))
    

    df = pd.DataFrame({
        'Epoch': epochs * 4,
        'Value': history['train_acc'] + history['val_acc'] + history['train_loss'] + history['val_loss'],
        'Metric': (['Accuracy'] * (epochs_count * 2)) + (['Loss'] * (epochs_count * 2)),
        'Type': (['Train'] * epochs_count + ['Val'] * epochs_count) * 2
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy
    sns.lineplot(
        data=df[df['Metric'] == 'Accuracy'], 
        x='Epoch', y='Value', hue='Type', 
        ax=axes[0], marker='o', palette='viridis'
    )
    axes[0].set_title('Training & Validation Accuracy', fontsize=14, pad=15)
    axes[0].set_ylabel('Accuracy (%)')

    # Loss
    sns.lineplot(
        data=df[df['Metric'] == 'Loss'], 
        x='Epoch', y='Value', hue='Type', 
        ax=axes[1], marker='o', palette='magma'
    )
    axes[1].set_title('Training & Validation Loss', fontsize=14, pad=15)
    axes[1].set_ylabel('Loss')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Wykres Seaborn zapisany w: {save_path}")
    
    plt.show()