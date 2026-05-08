import torch
import argparse
import sys
from src.utils.logger import Logger
from src.engine.trainer_cls import train_classification
from src.engine.trainer_seg import train_segmentation
from src.engine.trainer_joint import train_multitask
from src.engine.trainer_attention import train_attention
HYPERPARAMS = {
    "LR_CLF": 1e-4,
    "LR_SEG": 1e-3,
    "BATCH_CLF": 32,
    "BATCH_SEG": 12,
    "BATCH_JOINT": 12,
    "EPOCHS": 40,
    "OPTIMIZER": "AdamW",
    "LOSS_CLF": "CrossEntropyLoss",
    "LOSS_SEG": "BCEWithLogitsLoss (Weighted) + DiceLoss",
    "EARLY_STOP_PATIENCE": 10,
    "WEIGHT_DECAY": 1e-3
}

# Dataset stats from preprocessing
CLASS_COUNTS = {'glioma': 1401, 'meningioma': 1635, 'no_tumor': 1207, 'pituitary': 1757}
total_samples = sum(CLASS_COUNTS.values())
class_weights_list = [total_samples / (4 * CLASS_COUNTS[cls]) for cls in sorted(CLASS_COUNTS.keys())]
POS_WEIGHT = 5.0 #58.0
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train models") 
    parser.add_argument( "--task", type=str, required=True, 
                        choices=["clf", "seg","joint","attn"], 
                        help="Choose task: clf (classification) or seg (segmentation)" ) 
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_weights = torch.tensor(class_weights_list).to(DEVICE)
    pos_weight = torch.tensor([POS_WEIGHT]).to(DEVICE)
    
    log_file = f"logs/{args.task}_training_log.txt"
    sys.stdout = Logger(log_file)
    
    print(f"Running on {DEVICE}")
    print("HYPERPARAMETERS SUMMARY")
    for k, v in HYPERPARAMS.items():
        print(f"{k:<20}: {v}")

    if args.task == 'clf':
        train_classification(DEVICE=DEVICE,HYPERPARAMS=HYPERPARAMS,class_weights=class_weights)
    elif args.task == 'seg':
        train_segmentation(DEVICE=DEVICE,HYPERPARAMS=HYPERPARAMS,pos_weight=pos_weight)
    elif args.task == 'joint':
        train_multitask(DEVICE=DEVICE,HYPERPARAMS=HYPERPARAMS,class_weights=class_weights,pos_weight=pos_weight)
    else:
        train_attention(DEVICE=DEVICE,HYPERPARAMS=HYPERPARAMS,pos_weight=pos_weight)
