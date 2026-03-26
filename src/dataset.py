import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, WeightedRandomSampler

class HER2Dataset(Dataset):
    def __init__(self, data_path, transform=None):

        self.data_path = Path(data_path)
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        self.class_to_idx = {
            'class_0': 0,
            'class_1+': 1,
            'class_2+': 2,
            'class_3+': 3
        }
        
        for class_name, label in self.class_to_idx.items():
            class_dir = self.data_path / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        self.image_paths.append(str(img_path))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(img_size=224, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),      
            A.VerticalFlip(p=0.5),        
            A.RandomRotate90(p=0.5),      
            A.ShiftScaleRotate(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    


def get_dataloader(dataset, batch_size=32, is_train=True):
    if not is_train:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    target_labels = dataset.labels
    
    class_sample_count = np.array([len(np.where(target_labels == t)[0]) for t in np.unique(target_labels)])
    
    weight = 1. / class_sample_count
    
    samples_weight = np.array([weight[t] for t in target_labels])
    samples_weight = torch.from_numpy(samples_weight)
    
    sampler = WeightedRandomSampler(
        weights=samples_weight, 
        num_samples=len(samples_weight), 
        replacement=True
    )

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)