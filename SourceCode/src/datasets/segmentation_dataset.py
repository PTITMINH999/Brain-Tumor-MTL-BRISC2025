import torch
from torch.utils.data import Dataset
import numpy as np
import json 
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.samples = []

        img_dir = self.data_dir / 'segmentation' / split / 'images'
        mask_dir = self.data_dir / 'segmentation' / split / 'masks'

        if not img_dir.exists():
            print(f"{img_dir} does not exist")
            return

        for img_path in img_dir.glob('*.npy'):
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                self.samples.append((img_path, mask_path))
        
        if len(self.samples) == 0:
            print(f"No samples found in {img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = np.load(img_path)
        mask = np.load(mask_path)

        image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).permute(2, 0, 1).float()
            elif mask.ndim == 3 and mask.shape[-1] == 1: 
                mask = mask.permute(2, 0, 1).float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        return image, mask

# def get_segmentation_transforms(split='train'):
#     if split == 'train':
#         return A.Compose([
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.3),
#             A.Affine(scale=(0.85, 1.15), translate_percent=(-0.1, 0.1), rotate=(-25, 25), p=0.7),
#             A.ElasticTransform(alpha=1, sigma=50, p=0.3),
#             A.GridDistortion(p=0.3),
#             A.OpticalDistortion(distort_limit=0.1, p=0.3),
#             A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
#             A.RandomGamma(gamma_limit=(80, 120), p=0.3),
#             A.GaussNoise(p=0.3),
#             A.GaussianBlur(blur_limit=(3, 5), p=0.2),
#             ToTensorV2()
#         ])
#     else:
#         return A.Compose([ToTensorV2()])

def get_segmentation_transforms(split='train'):
    if split == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-20, 20), p=0.7),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            ToTensorV2()
        ])
    else:
        return A.Compose([ToTensorV2()])