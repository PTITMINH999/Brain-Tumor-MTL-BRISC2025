import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import os

from src.models.unetclassifier import UnetClassifier
from src.datasets.classification_dataset import ClassificationDataset, get_classification_transforms
from src.utils.early_stopping import EarlyStopping
from src.utils.plotting import plot_learning_curves
from src.engine.evaluator import evaluate_classification,save_confusion_matrix

def train_classification(DEVICE,HYPERPARAMS,class_weights):
    print("CLASSIFICATION TRAINING")

    
    train_ds = ClassificationDataset('data/processed', split='train', 
                                          transform=get_classification_transforms('train'))
    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_sub, val_sub = random_split(train_ds, [train_size, val_size], 
                                      generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_sub, batch_size=HYPERPARAMS['BATCH_CLF'], 
                             shuffle=True, num_workers=4, pin_memory=True, 
                             persistent_workers=True)
    val_loader = DataLoader(val_sub, batch_size=HYPERPARAMS['BATCH_CLF'], 
                           shuffle=False, num_workers=2, pin_memory=True, 
                           persistent_workers=True)

    model_clf = UnetClassifier(1, 4).to(DEVICE)
    optimizer = optim.AdamW(model_clf.parameters(), lr=HYPERPARAMS['LR_CLF'], 
                          weight_decay=HYPERPARAMS['WEIGHT_DECAY'])
    criterion_clf = nn.CrossEntropyLoss(weight=class_weights)
    early_stopping = EarlyStopping(patience=HYPERPARAMS['EARLY_STOP_PATIENCE'], mode='max')
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0

    for epoch in range(HYPERPARAMS['EPOCHS']):
        model_clf.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for img, label in tqdm(train_loader, desc=f"Clf Epoch {epoch+1}/{HYPERPARAMS['EPOCHS']}"):
            img, label = img.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            out = model_clf(img)
            loss = criterion_clf(out, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            _, pred = torch.max(out, 1)
            correct += (pred == label).sum().item()
            total += label.size(0)
        
        train_acc = correct / total
        
        model_clf.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                out = model_clf(img)
                val_loss += criterion_clf(out, label).item()
                _, pred = torch.max(out, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        print(f"  Epoch {epoch+1}: Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_clf.state_dict(), 'checkpoints/classification/unet_clf_best.pth')
            print(f" Saved best model (Val Acc: {val_acc:.4f})")
        
        if early_stopping(val_acc):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    plot_learning_curves(history, 'Classification', 'classification/classification_results.png')
    

    print("\nEVALUATING CLASSIFICATION ON TEST SET...")
    test_ds = ClassificationDataset('data/processed', split='test', 
                                         transform=get_classification_transforms('test'))
    test_loader = DataLoader(test_ds, batch_size=HYPERPARAMS['BATCH_CLF'], 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    if os.path.exists('checkpoints/classification/unet_clf_best.pth'):
        model_clf.load_state_dict(torch.load('checkpoints/classification/unet_clf_best.pth'))
    
    results = evaluate_classification(model_clf, test_loader, criterion_clf, DEVICE)
    print(f"Test Acc: {results['acc']:.4f} | Prec: {results['precision']:.4f} | "
          f"Recall: {results['recall']:.4f} | F1: {results['f1']:.4f}")
    save_confusion_matrix(results['cm'], 'Classification Test CM', 'classification/test_clf_cm.png')
    
    del train_loader, val_loader, test_loader, model_clf, optimizer
    torch.cuda.empty_cache()