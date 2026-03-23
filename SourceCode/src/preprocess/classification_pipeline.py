import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from src.preprocess.image_preprocess import preprocess_image
from configs.preprocess_config import DATA_DIR,PROCESSED_DIR,CONFIG

def run_classification():
    train_source = DATA_DIR / 'classification_task' / 'train'
    if not train_source.exists():
        print(f"Could not find {train_source}")
        return None

    classes = [d.name for d in train_source.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(classes))}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / 'class_mapping.json', 'w') as f:
        json.dump(class_to_idx, f)

    stats = {
        'total': 0, 
        'processed': 0, 
        'failed': 0, 
        'per_class': {cls: 0 for cls in classes},
        'splits': {'train': 0, 'test': 0}  # Track per-split counts
    }
    
    for split in ['train', 'test']:
        split_dir = DATA_DIR / 'classification_task' / split
        output_base = PROCESSED_DIR /'classification'/ split

        if not split_dir.exists(): 
            print("directory not found")
            continue

        for cls in classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists(): 
                continue

            (output_base / cls).mkdir(parents=True, exist_ok=True)
            
            files = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.jpeg')) + list(cls_dir.glob('*.png'))
            stats['total'] += len(files)
            stats['per_class'][cls] += len(files)
            stats['splits'][split] += len(files)

            for img_path in tqdm(files, desc=f"{split}/{cls}"):
                img = preprocess_image(
                    img_path, 
                    target_size=CONFIG['target_size'],
                    apply_clahe=CONFIG['apply_clahe'],
                    normalize=CONFIG['normalize'],
                    clahe_clip_limit=CONFIG['clahe_clip_limit'],
                    clahe_tile_size=CONFIG['clahe_tile_size'],
                    denoise=CONFIG['denoise']
                )
                if img is not None:
                    np.save(output_base / cls / (img_path.stem + '.npy'), img)
                    stats['processed'] += 1
                else:
                    stats['failed'] += 1
    
    print(f"\nClassification Stats: {stats['processed']}/{stats['total']} processed, {stats['failed']} failed")
    print(f"   Per-class counts: {stats['per_class']}")
    print(f"   Train/Test split: {stats['splits']}")
    
    if stats['per_class']:
        plt.figure(figsize=(10, 5))
        colors = ["#e73cd0", '#3498db', '#2ecc71', "#f3e012"]
        bars = plt.bar(stats['per_class'].keys(), stats['per_class'].values(), 
                       color=colors, edgecolor='black', linewidth=1.5)
        plt.title('Class Distribution (All Splits)', fontsize=14, fontweight='bold')
        plt.xlabel('Tumor Type', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add count labels
        for bar, (cls, count) in zip(bars, stats['per_class'].items()):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(PROCESSED_DIR / 'class_balance.png', dpi=300)
        plt.close()
        print(f"Saved class distribution chart")
    
    return stats
