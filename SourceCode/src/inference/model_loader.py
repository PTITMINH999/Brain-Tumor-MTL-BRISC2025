import torch
import numpy as np
import cv2
from configs.inference_config import DEVICE,CLASS_MAPPING_PATH,TARGET_SIZE,MODEL_CONFIGS
from pathlib import Path
import json
from src.models.unet import Unet  
from src.models.unetclassifier import UnetClassifier
from src.models.attentionunet import AttentionUNet
from src.models.multitask import UnetWithClassifier

MODEL = {}
def load_model(model_path, model_type):
    print(f"Loading {model_type} architecture on {DEVICE}...")
    
    if model_type == 'joint':
        model = UnetWithClassifier(n_channels=1, n_seg_classes=1, n_clf_classes=4)
    elif model_type == 'attention':
        model = AttentionUNet(n_channels=1, n_classes=1)
    elif model_type == 'unet':
        model = Unet(n_channels=1, n_classes=1)
    elif model_type == 'classifier':
        model = UnetClassifier(n_channels=1, n_classes=4)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"Loaded weights from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None

def load_all_models():
    global MODEL

    # nếu đã load rồi thì return luôn
    if MODEL:
        print("Models already loaded, reuse...")
        return MODEL
    print("Loading ALL models (only once)...")

    MODEL['joint'] = load_model(
        MODEL_CONFIGS['joint']['path'], 
        MODEL_CONFIGS['joint']['type']
    )

    MODEL['attention'] = load_model(
        MODEL_CONFIGS['attention']['path'], 
        MODEL_CONFIGS['attention']['type']
    )

    MODEL['unet'] = load_model(
        MODEL_CONFIGS['unet']['path'], 
        MODEL_CONFIGS['unet']['type']
    )

    MODEL['classifier'] = load_model(
        MODEL_CONFIGS['classifier']['path'], 
        MODEL_CONFIGS['classifier']['type']
    )

    print("All models loaded!")

    return MODEL
def load_class_mapping():
    try:
        with open(CLASS_MAPPING_PATH, 'r') as f:
            mapping = json.load(f)
        return {v: k for k, v in mapping.items()}
    except:
        return {0: 'glioma', 1: 'meningioma', 2: 'no_tumor', 3: 'pituitary'}

def preprocess_image(img_path):
    if isinstance(img_path, (str, Path)):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
    else:
        img = img_path
    
    img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = img.astype(np.float32) / 255.0
    return img