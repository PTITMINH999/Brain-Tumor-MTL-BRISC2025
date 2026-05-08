import torch
import numpy as np
from configs.inference_config import TARGET_SIZE,DEVICE

def predict(model, img_preprocessed, idx_to_class, model_type):
    img_tensor = torch.from_numpy(img_preprocessed).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    
    pred_class = "N/A"
    confidence = 0.0
    seg_mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    all_probs = [0.0] * 4

    with torch.no_grad():
        output = model(img_tensor)
        
        if model_type == 'joint':
            seg_logits, clf_logits = output
            seg_mask = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
            probs = torch.softmax(clf_logits, dim=1).squeeze().cpu().numpy()
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            pred_class = idx_to_class[pred_idx]
            all_probs = probs

        elif model_type in ['attention', 'unet']:
            seg_logits = output
            seg_mask = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
            pred_class = "Segmentation Only"
            confidence = 1.0
        
        elif model_type == 'classifier':
            clf_logits = output
            probs = torch.softmax(clf_logits, dim=1).squeeze().cpu().numpy()
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            pred_class = idx_to_class[pred_idx]
            all_probs = probs

    seg_mask_binary = (seg_mask > 0.5).astype(np.float32)
    return pred_class, confidence, seg_mask_binary, all_probs

def determine_best_model(all_results):
    best_model = None
    best_score = -1
    
    for model_key, result in all_results.items():
        if result['confidence'] > 0 and result['pred_class'] != "Segmentation Only":
            score = result['confidence']
        elif result['pred_class'] == "Segmentation Only":
            tumor_pct = result['tumor_percentage']
            score = 0.7 if tumor_pct > 0.01 else 0.3
        else:
            score = 0
        
        if score > best_score:
            best_score = score
            best_model = model_key
    
    return best_model