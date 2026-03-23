import torch
from torch.utils.data import Dataset
import numpy as np
import json 
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class NpyJointDataset(Dataset):
    def __init__(self,data_dir,split='train',transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = [] # luu tuple(img_path,mask_path,label)
        seg_img_dir = self.data_dir / 'segmentation' / split / 'images'
        seg_mask_dir = self.data_dir / 'segmentation' / split / 'masks'
        self.file_to_label = {} # mapping ten file : nhan classification

        with open(self.data_dir / 'class_mapping.json') as f:
            class_map = json.load(f)
        
        for cls_name, cls_idx in class_map.items():
            cls_path = self.data_dir /'classification'/ split / cls_name
            if cls_path.exists():
                for f in cls_path.glob('*.npy'):
                    self.file_to_label[f.name] = cls_idx
        
        for img_path in seg_img_dir.glob('*.npy'):
            mask_path = seg_mask_dir / img_path.name
            if img_path.name in self.file_to_label and mask_path.exists():
                self.samples.append((img_path, mask_path, self.file_to_label[img_path.name]))

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path,mask_path,label = self.samples[idx]
        img = np.load(img_path)
        mask = np.load(mask_path)
        img = np.expand_dims(img,axis=-1)
        mask = np.expand_dims(mask,axis=-1)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).permute(2, 0, 1).float()
            elif mask.shape[-1] == 1:
                mask = mask.permute(2, 0, 1).float()
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        
        return img, mask, torch.tensor(label, dtype=torch.long)







