import torch 
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.metrics.segmentation_metrics import calculate_seg_metrics
def evaluate_classification(model,test_loader,criterion,device):
    model.eval()
    test_loss = 0
    all_preds,all_labels = [],[]
    with torch.no_grad():
        for img,label in test_loader:
            img,label = img.to(device),label.to(device)
            res = model(img)
            test_loss += criterion(res,label).item()
            _,pred = torch.max(res,1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_acc = (all_preds == all_labels).mean()
    precision = precision_score(all_labels,all_preds,average='weighted',zero_division=0)
    recall = recall_score(all_labels,all_preds,average='weighted',zero_division=0)
    f1 = f1_score(all_labels,all_preds,average='weighted',zero_division=0)
    cm = confusion_matrix(all_labels,all_preds)

    return {
        'loss': test_loss / len(test_loader),
        'acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm
    }

def evaluate_segmentation(model,test_loader,bce,dice_loss_fn,device):
    model.eval()
    test_loss = 0
    total_iou,total_dice,total_acc = 0,0,0

    with torch.no_grad():
        for img,mask in test_loader:
            img,mask = img.to(device),mask.to(device)
            out = model(img)
            test_loss += (bce(out,mask) + dice_loss_fn(out,mask).item())
            pred = (torch.sigmoid(out) > 0.5).float()
            d,i,a = calculate_seg_metrics(pred,mask)
            total_dice += d
            total_iou += i
            total_acc += a 

    return{
        'loss': test_loss/len(test_loader),
        'iou': total_iou/len(test_loader),
        'dice':total_dice/len(test_loader),
        'pixel_acc': total_acc/len(test_loader)
    }

def evaluate_joint(model,test_loader,bce,dice_loss_fn,criterion_clf,device):
    model.eval()
    test_loss = 0
    total_iou,total_dice,total_seg_acc = 0,0,0
    all_preds,all_labels = [],[]

    with torch.no_grad():
        for img,mask,label in test_loader:
            img,mask,label = img.to(device),mask.to(device),label.to(device)
            seg_output,clf_output = model(img)

            seg_loss = bce(seg_output,mask) + dice_loss_fn(seg_output,mask)
            clf_loss = criterion_clf(clf_output,label)
            test_loss += (seg_loss + clf_loss).item()

            pred_mask = (torch.sigmoid(seg_output) > 0.5).float()
            d,i,a = calculate_seg_metrics(pred_mask,mask)
            total_iou += i
            total_dice += d
            total_seg_acc += a

            _,pred_clf = torch.max(clf_output,1)
            all_preds.extend(pred_clf.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    clf_acc = (all_preds==all_labels).mean()

    return {
        'loss': test_loss / len(test_loader),
        'clf_acc': clf_acc,
        'iou': total_iou / len(test_loader),
        'dice': total_dice / len(test_loader),
        'pixel_acc': total_seg_acc / len(test_loader),
        'cm': confusion_matrix(all_labels, all_preds)
    }


def save_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/{filename}', dpi=300)
    plt.close()