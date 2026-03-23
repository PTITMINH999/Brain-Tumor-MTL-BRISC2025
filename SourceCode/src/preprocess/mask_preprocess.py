import os
import numpy as np
import cv2

def preprocess_mask(mask_path, target_size=224, adaptive_threshold=False):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None: 
        print(f"Could not read mask {mask_path}")
        return None
    
    # Resize with NEAREST to preserve binary values
    mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    

    if adaptive_threshold:
        mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        mask = (mask > 127).astype(np.float32)
    else:
        mask = (mask > 127).astype(np.float32) 
    
    return mask