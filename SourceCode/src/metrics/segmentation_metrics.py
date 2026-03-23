import torch 

def calculate_seg_metrics(pred_mask,true_mask):
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)
    intersection = (pred_mask * true_mask).sum()
    total = pred_mask.sum() + true_mask.sum()
    union = total - intersection
    dice = (2. * intersection + 1e-6) / (total + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    correct = (pred_mask == true_mask).sum()
    pixel_acc = correct / (len(true_mask) + 1e-6)
    return dice.item(),iou.item(),pixel_acc.item()

