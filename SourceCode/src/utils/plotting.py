from pathlib import Path 
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curves(history, title, filename):
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='Train Loss', color='#e74c3c')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss', color="#3445db")
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if 'train_acc' in history:
        plt.plot(history['train_acc'], label='Train Acc', color='#2ecc71', linestyle='--')
    if 'val_acc' in history:
        plt.plot(history['val_acc'], label='Val Acc', color='#9b59b6')
    if 'val_iou' in history: 
        plt.plot(history['val_iou'], label='Val mIoU', color='#f39c12', linewidth=2)
    if 'val_dice' in history: 
        plt.plot(history['val_dice'], label='Val Dice', color="#44ad9f", linestyle='-.')
    if 'val_pixel_acc' in history: 
        plt.plot(history['val_pixel_acc'], label='Val Pixel Acc', color="#a09e16", linestyle=':')    

        

    
    plt.title(f'{title} - Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{filename}', dpi=300)
    plt.close()