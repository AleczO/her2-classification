import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch

sns.set_theme(style="whitegrid", palette="muted")

def plot_distribution(data_input, title="HER2 Class Distribution"):
    """
    Plots the distribution of classes. 
    Handles both a dictionary of counts OR a raw PyTorch Dataset.
    """

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # CASE 1: Input is a dictionary (e.g., {'0': 100, '1+': 150...})
    if isinstance(data_input, dict):
        # Convert dict to DataFrame for Seaborn
        df = pd.DataFrame(list(data_input.items()), columns=['Class', 'Count'])
        # Sort by class name to keep order 0, 1+, 2+, 3+
        df = df.sort_values('Class')
        ax = sns.barplot(data=df, x='Class', y='Count', palette="viridis", hue='Class', legend=False)
    
    # CASE 2: Input is a Dataset (list of tuples like (image, label))
    else:
        labels = [label for _, label in data_input]
        ax = sns.countplot(x=labels, palette="viridis", hue=labels, legend=False)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("HER2 Grade", fontsize=12)
    plt.ylabel("Number of Patches", fontsize=12)

    # Add count labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points',
                        fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()



def plot_image_gallery(dataset, samples_per_class=1, title="HER2 Class Samples"):
    """
    Displays a fixed 2x2 grid with one image for each class (0, 1+, 2+, 3+).
    """
    
    # 1. Find the first index for each class from the labels list
    labels = np.array(dataset.labels)
    indices = [np.where(labels == i)[0][0] for i in range(4)]
    
    # 2. Visual setup (ImageNet stats to reverse normalization)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    class_names = ["0", "1+", "2+", "3+"]

    # 3. Iterate through indices and render the grid
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        
        # Convert tensor to numpy and denormalize
        img = img.permute(1, 2, 0).numpy() * std + mean
        img = np.clip(img, 0, 1)
        
        ax = axes[i // 2, i % 2]
        ax.imshow(img)
        ax.set_title(f"HER2 Score: {class_names[i]}", fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.suptitle(title, fontsize=18, y=0.95)
    plt.tight_layout()
    plt.show()