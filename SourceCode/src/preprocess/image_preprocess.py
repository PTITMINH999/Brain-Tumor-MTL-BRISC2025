import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

def preprocess_image(img_path, target_size=224, apply_clahe=True, normalize=True, 
                     clahe_clip_limit=2.0, clahe_tile_size=(8, 8), denoise=False):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None: 
        print(f"Warning: Could not read {img_path}")
        return None
    
    # Validate image dimensions
    if img.shape[0] < 50 or img.shape[1] < 50:
        print(f"Warning: Image {img_path.name} too small ({img.shape}), skipping")
        return None
    
    # Denoise before resizing
    if denoise:
        img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Resize
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Apply CLAHE
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size)
        img = clahe.apply(img)
    
    # Normalize to [0, 1]
    if normalize:
        img = img.astype(np.float32) / 255.0
    
    return img