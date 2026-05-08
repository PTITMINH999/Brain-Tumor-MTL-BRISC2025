import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
import os

from src.models.multitask import UnetWithClassifier
from src.losses.dice_loss import DiceLoss
from src.metrics.segmentation_metrics import calculate_seg_metrics
from src.datasets.joint_dataset import NpyJointDataset    
from src.datasets.segmentation_dataset import get_segmentation_transforms               
from src.utils.early_stopping import EarlyStopping
from src.utils.plotting import plot_learning_curves
from src.engine.evaluator import evaluate_joint,save_confusion_matrix

def train_multitask(DEVICE,HYPERPARAMS,class_weights,pos_weight):
    print("JOINT TRAINING")
    full_joint_ds = NpyJointDataset('data/processed', split='train', 
                                    transform=get_segmentation_transforms('train'))
    train_size = int(0.8 * len(full_joint_ds))
    val_size = len(full_joint_ds) - train_size
    train_ds, val_ds = random_split(full_joint_ds, [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=HYPERPARAMS['BATCH_JOINT'], 
                             shuffle=True, num_workers=4, pin_memory=True, 
                             persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=HYPERPARAMS['BATCH_JOINT'], 
                           shuffle=False, num_workers=2, pin_memory=True, 
                           persistent_workers=True)

    model_joint = UnetWithClassifier(1, 1, 4).to(DEVICE)
    optimizer = optim.AdamW(model_joint.parameters(), lr=HYPERPARAMS['LR_SEG'], 
                          weight_decay=HYPERPARAMS['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    crit_clf = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss_fn = DiceLoss()
    early_stopping = EarlyStopping(patience=HYPERPARAMS['EARLY_STOP_PATIENCE'], mode='max')
    
    history_joint = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 
        'val_acc': [], 'val_iou': [], 'val_dice': [], 'val_pixel_acc': []
    }
    best_score = 0

    for epoch in range(HYPERPARAMS['EPOCHS']):
        model_joint.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for img, mask, label in tqdm(train_loader, desc=f"Joint Epoch {epoch+1}/{HYPERPARAMS['EPOCHS']}"):
            img, mask, label = img.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            s, c = model_joint(img)
            loss = (bce(s, mask) + dice_loss_fn(s, mask)) + crit_clf(c, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            _, pc = torch.max(c, 1)
            correct += (pc == label).sum().item()
            total += label.size(0)

        train_acc = correct / total
        
        # Validation
        model_joint.eval()
        val_loss_sum = 0
        val_iou, val_dice, val_seg_acc, val_clf_acc = 0, 0, 0, 0
        
        with torch.no_grad():
            for img, mask, label in val_loader:
                img, mask, label = img.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
                s, c = model_joint(img)
                val_loss_sum += ((bce(s, mask) + dice_loss_fn(s, mask)) + crit_clf(c, label)).item()
                
                p = (torch.sigmoid(s) > 0.5).float()
                d, i, a = calculate_seg_metrics(p, mask)
                val_iou += i
                val_dice += d
                val_seg_acc += a
                
                _, pc = torch.max(c, 1)
                val_clf_acc += (pc == label).sum().item() / label.size(0)

        avg_clf_acc = val_clf_acc / len(val_loader)
        avg_iou = val_iou / len(val_loader)
        avg_dice = val_dice / len(val_loader)
        avg_seg_acc = val_seg_acc / len(val_loader)

        
        history_joint['train_loss'].append(epoch_loss / len(train_loader))
        history_joint['train_acc'].append(train_acc)
        history_joint['val_loss'].append(val_loss_sum / len(val_loader))
        history_joint['val_acc'].append(avg_clf_acc)
        history_joint['val_iou'].append(avg_iou)
        history_joint['val_dice'].append(avg_dice)
        history_joint['val_pixel_acc'].append(avg_seg_acc)
        
        print(f"  Epoch {epoch+1}: Clf Acc={avg_clf_acc:.4f} | mIoU={avg_iou:.4f} | "
              f"Dice={avg_dice:.4f} | PixAcc={avg_seg_acc:.4f}")
        
        score = 0.6 * avg_iou + 0.4 * avg_clf_acc
        scheduler.step(score)
        
        if score > best_score:
            best_score = score
            torch.save(model_joint.state_dict(), 'checkpoints/joint/unet_joint_best.pth')
            print(f"Saved best model (Combined Score: {score:.4f})")
        
        if early_stopping(score):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    plot_learning_curves(history_joint, 'Joint Training', 'joint/joint_metrics.png')
    

    print("\nGenerating Joint Model Confusion Matrix...")
    if os.path.exists('checkpoints/joint/unet_joint_best.pth'):
        model_joint.load_state_dict(torch.load('checkpoints/joint/unet_joint_best.pth'))
    
    model_joint.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for img, mask, label in val_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            _, c = model_joint(img)
            _, pred = torch.max(c, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    save_confusion_matrix(cm, 'Joint Model Validation CM', 'joint/joint_val_cm.png')
    
    # Test Evaluation
    print("\nEVALUATING JOINT MODEL ON TEST SET...")
    test_ds = NpyJointDataset('data/processed', split='test', 
                              transform=get_segmentation_transforms('test'))
    test_loader = DataLoader(test_ds, batch_size=HYPERPARAMS['BATCH_JOINT'], 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    results = evaluate_joint(model_joint, test_loader, bce, dice_loss_fn, crit_clf, DEVICE)
    print("=== Classification ===")
    print(
        f"Acc: {results['clf_acc']:.4f} | "
        f"Prec: {results['precision']:.4f} | "
        f"Recall: {results['recall']:.4f} | "
        f"F1: {results['f1']:.4f}"
    )

    print("=== Segmentation ===")
    print(
        f"mIoU: {results['iou']:.4f} | "
        f"Dice: {results['dice']:.4f} | "
        f"Pixel Acc: {results['pixel_acc']:.4f}"
    )   
    save_confusion_matrix(results['cm'], 'Joint Model Test CM', 'joint/joint_test_cm.png')
    
    del train_loader, val_loader, test_loader, model_joint, optimizer
    torch.cuda.empty_cache()

















