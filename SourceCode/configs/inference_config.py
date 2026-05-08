import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_MAPPING_PATH = 'data/processed/class_mapping.json'
TARGET_SIZE = 224

MODEL_CONFIGS = {
    'joint': {
        'path': 'checkpoints/joint/unet_joint_best.pth',
        'type': 'joint',
        'name': 'Joint Model'
    },
    'attention': {
        'path': 'checkpoints/segmentation/att_unet_best.pth',
        'type': 'attention',
        'name': 'Attention U-Net'
    },
    'unet': {
        'path': 'checkpoints/segmentation/unet_seg_best.pth',
        'type': 'unet',
        'name': 'Base U-Net'
    },
    'classifier': {
        'path': 'checkpoints/classification/unet_clf_best.pth',
        'type': 'classifier',
        'name': 'Classifier'
    }
}