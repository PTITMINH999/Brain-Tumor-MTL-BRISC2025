import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from src.preprocess.classification_pipeline import run_classification
from src.preprocess.segmentation_pipeline import run_segmentation
from configs.preprocess_config import DATA_DIR,PROCESSED_DIR,CONFIG
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

def validate_preprocessing():
    
    validation_results = {'status': 'success', 'checks': []}
    
    # Check classification
    class_mapping_path = PROCESSED_DIR / 'class_mapping.json'
    if class_mapping_path.exists():
        with open(class_mapping_path) as f:
            class_map = json.load(f)
        print(f"Class mapping: {class_map}")
        validation_results['checks'].append(('class_mapping', 'pass', class_map))
    else:
        print("Class mapping not found")
        validation_results['checks'].append(('class_mapping', 'fail', None))
        validation_results['status'] = 'failed'
    
    # Check a sample image
    sample_dirs = list((PROCESSED_DIR /'classification'/ 'train').glob('*/*.npy'))
    if sample_dirs:
        sample = np.load(sample_dirs[5])
        print(f"Sample classification image shape: {sample.shape}")
        print(f"Value range: [{sample.min():.3f}, {sample.max():.3f}]")
        validation_results['checks'].append(('sample_image', 'pass', {
            'shape': sample.shape,
            'min': float(sample.min()),
            'max': float(sample.max())
        }))
    else:
        print("No classification samples found")
        validation_results['checks'].append(('sample_image', 'warn', None))
    
    # Check segmentation
    seg_imgs = list((PROCESSED_DIR / 'segmentation' / 'train' / 'images').glob('*.npy'))
    seg_masks = list((PROCESSED_DIR / 'segmentation' / 'train' / 'masks').glob('*.npy'))
    print(f"Segmentation images: {len(seg_imgs)}, masks: {len(seg_masks)}")
    
    if seg_imgs and seg_masks:
        img = np.load(seg_imgs[5])
        mask = np.load(seg_masks[5])
        print(f"   Image shape: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"   Mask shape: {mask.shape}, unique values: {np.unique(mask)}, foreground ratio: {np.mean(mask):.4f}")
        validation_results['checks'].append(('segmentation', 'pass', {
            'num_images': len(seg_imgs),
            'num_masks': len(seg_masks),
            'image_shape': img.shape,
            'mask_unique_values': np.unique(mask).tolist(),
            'mask_foreground_ratio': float(np.mean(mask))
        }))
    else:
        print("No segmentation samples found")
        validation_results['checks'].append(('segmentation', 'warn', None))
    
    return validation_results

if __name__ == '__main__':
    clf_stats = run_classification()
    seg_stats = run_segmentation()
    val_results = validate_preprocessing()
    

    print("Preprocessing Complete!")

    preprocessing_stats = {
        'config': CONFIG,
        'classification': clf_stats if clf_stats else {},
        'segmentation': seg_stats if seg_stats else {},
        'validation': val_results
    }
    
    stats_path = PROCESSED_DIR / 'preprocessing_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(preprocessing_stats, f, indent=2, cls=NpEncoder)
    
    print(f"\nComplete preprocessing statistics saved to:")
    print(f"   {stats_path}")
    print(f"\nVisualization saved to:")
    print(f"{PROCESSED_DIR / 'class_balance.png'}")
    

    if clf_stats:
        print(f"\nSUMMARY:")
        print(f"   Classification: {clf_stats['processed']}/{clf_stats['total']} images")
        print(f"   Segmentation: {seg_stats['processed']}/{seg_stats['total']} images")