import torch
import numpy as np
import cv2
from configs.inference_config import MODEL_CONFIGS
from src.inference.model_loader import preprocess_image,load_class_mapping,load_model,MODEL
from src.inference.mri_view import detect_mri_view
from src.inference.predictor import predict,determine_best_model
from src.inference.visualization import (visualize_all_results,compare_models_overlay)
from pathlib import Path
import json

def run_all_models_inference(image_path, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    print(f"Processing image: {image_path}")

    img_original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    print("Detecting MRI view type")
    mri_view, view_conf = detect_mri_view(img_original)
    print(f"   Detected View: {mri_view} (Confidence: {view_conf:.1%})")
    
    img_prep = preprocess_image(image_path)
    img_vis = img_original.copy()
    idx_to_class = load_class_mapping()
    
    all_results = {}
    
    for model_key, config in MODEL_CONFIGS.items():
        model_path = Path(config['path'])
        
        if not model_path.exists():
            print(f"  {config['name']} not found at {model_path}, skipping...")
            continue
        
        print(f"Running {config['name']}...")
        
        model = load_model(model_path, config['type'])
        if model is None:
            continue
        
        pc, conf, mask, probs = predict(model, img_prep, idx_to_class, config['type'])
        
        all_results[model_key] = {
            'pred_class': pc,
            'confidence': conf,
            'seg_mask': mask,
            'all_probs': probs,
            'tumor_percentage': np.mean(mask)
        }
        
        print(f"Prediction: {pc}")
        if config['type'] not in ['attention', 'unet']:
            print(f"Confidence: {conf:.2%}")
        if config['type'] != 'classifier':
            print(f"Tumor Region: {np.mean(mask):.2%} coverage")
    
    if not all_results:
        print("\nNo models could be loaded. Please check model paths.")
        return
    
    best_model = determine_best_model(all_results)
    print(f"BEST MODEL: {MODEL_CONFIGS[best_model]['name']}")
    mask_path = (Path(image_path).parent.parent/ "mask"/ f"{Path(image_path).stem}.png")

    if mask_path.exists():
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask,(224, 224),interpolation=cv2.INTER_NEAREST)
        gt_mask = (gt_mask > 127).astype(np.float32)
        overlay_save_path = (
            Path(output_folder)
            / f"{Path(image_path).stem}_overlay.png"
        )
        compare_models_overlay(
            img_original=img_vis,
            gt_mask=gt_mask,
            all_results=all_results,
            save_path=overlay_save_path
        )
    else:
        print(f"Ground truth mask not found: {mask_path}")
    
    save_path = Path(output_folder) / f"{Path(image_path).stem}_all_models_result.png"
    visualize_all_results(img_original, all_results, idx_to_class, best_model, mri_view, view_conf, save_path)
    
    summary = {
        'image': str(image_path),
        'mri_view': {
            'type': mri_view,
            'confidence': float(view_conf)
        },
        'best_model': {
            'name': MODEL_CONFIGS[best_model]['name'],
            'key': best_model
        },
        'results': {}
    }
    
    for model_key, result in all_results.items():
        summary['results'][model_key] = {
            'model_name': MODEL_CONFIGS[model_key]['name'],
            'prediction': result['pred_class'],
            'confidence': float(result['confidence']),
            'tumor_percentage': float(result['tumor_percentage']),
            'is_best': (model_key == best_model)
        }
    
    json_path = Path(output_folder) / f"{Path(image_path).stem}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {json_path}")

def run_flask_inference(
    image_path,
    output_folder,
    gt_mask_path=None
):

    Path(output_folder).mkdir(
        parents=True,
        exist_ok=True
    )

    img_original = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if img_original is None:
        raise ValueError(
            f"Cannot read image: {image_path}"
        )

    mri_view, view_conf = detect_mri_view(
        img_original
    )

    img_prep = preprocess_image(image_path)

    idx_to_class = load_class_mapping()

    all_results = {}

    # inference
    for model_key, config in MODEL_CONFIGS.items():

        model = MODEL.get(model_key)

        if model is None:
            continue

        pc, conf, mask, probs = predict(
            model,
            img_prep,
            idx_to_class,
            config['type']
        )

        all_results[model_key] = {
            "pred_class": pc,
            "confidence": conf,
            "seg_mask": mask,
            "all_probs": probs,
            "tumor_percentage": float(np.mean(mask))
        }

    if not all_results:
        raise ValueError("No model loaded")

    best_model = determine_best_model(
        all_results
    )

    # =========================
    # FULL RESULT IMAGE
    # =========================

    result_filename = (
        f"{Path(image_path).stem}_result.png"
    )

    result_save_path = (
        Path(output_folder)
        / result_filename
    )

    visualize_all_results(
        img_original=img_original,
        all_results=all_results,
        idx_to_class=idx_to_class,
        best_model=best_model,
        mri_view=mri_view,
        view_conf=view_conf,
        save_path=result_save_path
    )

    # =========================
    # OVERLAY
    # =========================

    overlay_filename = (
        f"{Path(image_path).stem}_overlay.png"
    )

    overlay_save_path = (
        Path(output_folder)
        / overlay_filename
    )

    overlay_exists = False

    if gt_mask_path is not None:

        print("READING GT MASK:", gt_mask_path)

        gt_mask = cv2.imread(
            str(gt_mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if gt_mask is not None:

            gt_mask = cv2.resize(
                gt_mask,
                (
                    224,
                    224
                ),
                interpolation=cv2.INTER_NEAREST
            )

            gt_mask = (
                gt_mask > 127
            ).astype(np.uint8)

            compare_models_overlay(
                img_original=img_original,
                gt_mask=gt_mask,
                all_results=all_results,
                save_path=overlay_save_path
            )

            overlay_exists = overlay_save_path.exists()

            print(
                "OVERLAY CREATED:",
                overlay_exists
            )

        else:
            print("FAILED TO READ GT MASK")

    # =========================
    # SUMMARY
    # =========================

    summary = {
        "image": str(image_path),

        "mri_view": {
            "type": mri_view,
            "confidence": float(view_conf)
        },

        "best_model": {
            "name": MODEL_CONFIGS[best_model]["name"],
            "key": best_model
        },

        "results": {}
    }

    for model_key, result in all_results.items():

        summary["results"][model_key] = {

            "model_name":
                MODEL_CONFIGS[model_key]["name"],

            "prediction":
                result["pred_class"],

            "confidence":
                float(result["confidence"]),

            "tumor_percentage":
                float(result["tumor_percentage"]),

            "is_best":
                (model_key == best_model)
        }

    json_filename = (
        f"{Path(image_path).stem}_summary.json"
    )

    json_path = (
        Path(output_folder)
        / json_filename
    )

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {

        "image_url":
            f"/static/results/{result_filename}",

        "overlay_url":
            (
                f"/static/results/{overlay_filename}"
                if overlay_exists
                else f"/static/results/{result_filename}"
            ),

        "summary":
            summary
    }