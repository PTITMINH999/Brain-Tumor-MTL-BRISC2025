import numpy as np

def detect_mri_view(img):
    h, w = img.shape
    if h < 10 or w < 10:
        return "Unknown", 0.5
    
    aspect_ratio = w / h
    h_start, h_end = max(0, h//4), min(h, 3*h//4)
    w_start, w_end = max(0, w//4), min(w, 3*w//4)
    
    if h_end <= h_start or w_end <= w_start:
        return "Unknown", 0.5
    
    center_region = img[h_start:h_end, w_start:w_end]
    edge_parts = []
    if h//4 > 0:
        edge_parts.append(img[0:h//4, :].flatten())
    if h - 3*h//4 > 0:
        edge_parts.append(img[3*h//4:, :].flatten())
    if w//4 > 0:
        edge_parts.append(img[:, 0:w//4].flatten())
    if w - 3*w//4 > 0:
        edge_parts.append(img[:, 3*w//4:].flatten())
    
    if not edge_parts:
        edge_region = img.flatten()
    else:
        edge_region = np.concatenate(edge_parts)
    
    center_mean = np.mean(center_region)
    edge_mean = np.mean(edge_region)
    center_std = np.std(center_region)
    mid_w = w // 2
    if mid_w > 0:
        left_half = img[:, :mid_w]
        right_half = np.fliplr(img[:, w-mid_w:])
        if left_half.shape == right_half.shape:
            try:
                horizontal_symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
                if np.isnan(horizontal_symmetry):
                    horizontal_symmetry = 0.0
            except:
                horizontal_symmetry = 0.0
        else:
            horizontal_symmetry = 0.0
    else:
        horizontal_symmetry = 0.0
        
    mid_h = h // 2
    if mid_h > 0:
        top_half = img[:mid_h, :]
        bottom_half = np.flipud(img[h-mid_h:, :])
        
        if top_half.shape == bottom_half.shape:
            try:
                vertical_symmetry = np.corrcoef(top_half.flatten(), bottom_half.flatten())[0, 1]
                if np.isnan(vertical_symmetry):
                    vertical_symmetry = 0.0
            except:
                vertical_symmetry = 0.0
        else:
            vertical_symmetry = 0.0
    else:
        vertical_symmetry = 0.0
    
    scores = {'Axial': 0, 'Coronal': 0, 'Sagittal': 0}
    

    if horizontal_symmetry > 0.7 and aspect_ratio > 0.85 and aspect_ratio < 1.15:
        scores['Axial'] += 3
    if center_mean > edge_mean * 1.2:
        scores['Axial'] += 2
    
    if horizontal_symmetry > 0.6 and aspect_ratio < 0.9:
        scores['Coronal'] += 3
    if vertical_symmetry < 0.6:
        scores['Coronal'] += 1

    if horizontal_symmetry < 0.5:
        scores['Sagittal'] += 4
    if aspect_ratio > 0.8 and aspect_ratio < 1.2 and center_std > 0.15:
        scores['Sagittal'] += 2
    
    best_view = max(scores, key=scores.get)
    max_score = scores[best_view]
    total_score = sum(scores.values())
    confidence = max_score / max(total_score, 1) if total_score > 0 else 0.33

    if confidence < 0.4:
        best_view = "Unknown"
        confidence = 0.5
    
    return best_view, confidence