import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from src.preprocess.mask_preprocess import preprocess_mask
from src.preprocess.image_preprocess import preprocess_image
from configs.preprocess_config import DATA_DIR,PROCESSED_DIR,CONFIG

def run_segmentation():
    stats = {
        'total': 0, 
        'processed': 0, 
        'failed': 0, 
        'missing_mask': 0, 
        'fg_pixels': 0, 
        'total_pixels': 0,
        'splits': {'train': 0, 'test': 0}
    }
    
    for split in ['train', 'test']:
        img_dir = DATA_DIR / 'segmentation_task' / split / 'images'
        mask_dir = DATA_DIR / 'segmentation_task' / split / 'masks'

        out_img_dir = PROCESSED_DIR / 'segmentation' / split / 'images'
        out_mask_dir = PROCESSED_DIR / 'segmentation' / split / 'masks'

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists(): 
            print(f"Skipping {split} - directory not found")
            continue

        files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.png'))
        stats['total'] += len(files)
        print(f"Found {len(files)} images in {split}")

        for img_path in tqdm(files, desc=f"Segmentation {split}"):
            mask_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_mask = mask_dir / (img_path.stem + ext)
                if potential_mask.exists():
                    mask_path = potential_mask
                    break
            
            if mask_path is None:
                print(f"Warning: No mask found for {img_path.name}")
                stats['missing_mask'] += 1
                continue
            
            img = preprocess_image(
                img_path,
                target_size=CONFIG['target_size'],
                apply_clahe=CONFIG['apply_clahe'],
                normalize=CONFIG['normalize'],
                clahe_clip_limit=CONFIG['clahe_clip_limit'],
                clahe_tile_size=CONFIG['clahe_tile_size'],
                denoise=CONFIG['denoise']
            )
            mask = preprocess_mask(mask_path, target_size=CONFIG['target_size'], 
                                   adaptive_threshold=CONFIG['adaptive_threshold_masks'])
            
            if img is not None and mask is not None:
                np.save(out_img_dir / (img_path.stem + '.npy'), img)
                np.save(out_mask_dir / (img_path.stem + '.npy'), mask)
                stats['processed'] += 1
                stats['splits'][split] += 1
                stats['fg_pixels'] += np.sum(mask > 0)
                stats['total_pixels'] += mask.size
            else:
                stats['failed'] += 1
    
    print(f"\nSegmentation Stats: {stats['processed']}/{stats['total']} processed, "
          f"{stats['failed']} failed, {stats['missing_mask']} missing masks")
    print(f"   Train/Test split: {stats['splits']}")
    

    if stats['total_pixels'] > 0:
        fg_ratio = stats['fg_pixels'] / stats['total_pixels']
        pos_weight_suggest = (stats['total_pixels'] - stats['fg_pixels']) / stats['fg_pixels']
        stats['foreground_ratio'] = fg_ratio
        stats['pos_weight_suggestion'] = pos_weight_suggest
        print(f"Foreground pixel ratio: {fg_ratio:.4f}")
        print(f"Suggested pos_weight for BCE: {pos_weight_suggest:.2f}")
    
    return stats