import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
import os

from src.models.attentionunet import AttentionUNet
from src.losses.dice_loss import DiceLoss
from src.metrics.segmentation_metrics import calculate_seg_metrics
from src.datasets.segmentation_dataset import SegmentationDataset,get_segmentation_transforms
from src.utils.early_stopping import EarlyStopping
from src.utils.plotting import plot_learning_curves
from src.engine.evaluator import evaluate_segmentation

def train_attention(DEVICE,HYPERPARAMS,pos_weight):
    print("ATTENTION U-NET TRAINING")

    
    # Reuse segmentation dataset
    train_ds = SegmentationDataset('data/processed', split='train', 
                                        transform=get_segmentation_transforms('train'))
    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_sub, val_sub = random_split(train_ds, [train_size, val_size], 
                                      generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_sub, batch_size=HYPERPARAMS['BATCH_SEG'], 
                             shuffle=True, num_workers=4, pin_memory=True, 
                             persistent_workers=True)
    val_loader = DataLoader(val_sub, batch_size=HYPERPARAMS['BATCH_SEG'], 
                           shuffle=False, num_workers=2, pin_memory=True, 
                           persistent_workers=True)

    model_attn = AttentionUNet(1, 1).to(DEVICE)
    optimizer = optim.AdamW(model_attn.parameters(), lr=HYPERPARAMS['LR_SEG'], 
                          weight_decay=HYPERPARAMS['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dice_loss_fn = DiceLoss()
    early_stopping = EarlyStopping(patience=HYPERPARAMS['EARLY_STOP_PATIENCE'], mode='max')
    
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': [], 'val_pixel_acc': []}
    best_iou = 0

    for epoch in range(HYPERPARAMS['EPOCHS']):
        # Training
        model_attn.train()
        epoch_loss = 0
        
        for img, mask in tqdm(train_loader, desc=f"Attn Epoch {epoch+1}/{HYPERPARAMS['EPOCHS']}"):
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            out = model_attn(img)
            loss = bce(out, mask) + dice_loss_fn(out, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model_attn.eval()
        val_loss_sum = 0
        val_iou, val_dice, val_acc = 0, 0, 0
        
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                out = model_attn(img)
                val_loss_sum += (bce(out, mask) + dice_loss_fn(out, mask)).item()
                pred = (torch.sigmoid(out) > 0.5).float()
                d, i, a = calculate_seg_metrics(pred, mask)
                val_iou += i
                val_dice += d
                val_acc += a

        avg_iou = val_iou / len(val_loader)
        avg_dice = val_dice / len(val_loader)
        avg_acc = val_acc / len(val_loader)
        
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_loss'].append(val_loss_sum / len(val_loader))
        history['val_iou'].append(avg_iou)
        history['val_dice'].append(avg_dice)
        history['val_pixel_acc'].append(avg_acc)
        
        print(f"Epoch {epoch+1}: Val mIoU={avg_iou:.4f} | Dice={avg_dice:.4f} | "
              f"Pixel Acc={avg_acc:.4f}")
        
        scheduler.step(avg_iou)
        
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model_attn.state_dict(), 'checkpoints/segmentation/att_unet_best.pth')
            print(f"Saved best model (Val mIoU: {avg_iou:.4f})")
        
        if early_stopping(avg_iou):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    plot_learning_curves(history, 'Attention U-Net', 'segmentation/attention_metrics.png')
    

    print("\nEVALUATING ATTENTION U-NET ON TEST SET...")
    test_ds = SegmentationDataset('data/processed', split='test', 
                                       transform=get_segmentation_transforms('test'))
    test_loader = DataLoader(test_ds, batch_size=HYPERPARAMS['BATCH_SEG'], 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    if os.path.exists('checkpoints/segmentation/att_unet_best.pth'):
        model_attn.load_state_dict(torch.load('checkpoints/segmentation/att_unet_best.pth'))
    
    results = evaluate_segmentation(model_attn, test_loader, bce, dice_loss_fn, DEVICE)
    print(f"Test mIoU: {results['iou']:.4f} | Dice: {results['dice']:.4f} | "
          f"Pixel Acc: {results['pixel_acc']:.4f}")
    

    del train_loader, val_loader, test_loader, model_attn, optimizer
    torch.cuda.empty_cache()