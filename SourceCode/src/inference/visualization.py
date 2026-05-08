import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from configs.inference_config import MODEL_CONFIGS

def visualize_all_results(img_original, all_results, idx_to_class, best_model, mri_view, view_conf, save_path=None):
    n_models = len(all_results)
    fig = plt.figure(figsize=(12, 4 * n_models))
    
    view_color = 'blue' if view_conf > 0.6 else 'orange' if view_conf > 0.4 else 'red'
    view_status = 'OK' if view_conf > 0.6 else 'WARN' if view_conf > 0.4 else 'BAD'
    
    fig.suptitle(f'{view_status} MRI View: {mri_view} | Confidence: {view_conf:.1%}', 
                 fontsize=16, fontweight='bold', color=view_color, y=0.995)
    
    view_descriptions = {
        'Axial': 'Axial-Horizontal slice (top-down view)',
        'Coronal': 'Coronal-Frontal slice (front-back view)',
        'Sagittal': 'Sagittal-Side slice (left-right view)',
        'Unknown': 'Unknown-View type could not be determined'
    }
    view_desc = view_descriptions.get(mri_view, '')
    fig.text(0.5, 0.97, view_desc, ha='center', fontsize=11, style='italic', color='gray')
    
    row = 0
    for model_key, result in all_results.items():
        is_best = (model_key == best_model)
        border_color = 'blue' if is_best else 'gray'
        title_prefix = 'BEST: ' if is_best else ''
        
        model_name = MODEL_CONFIGS[model_key]['name']
        pred_class = result['pred_class']
        confidence = result['confidence']
        seg_mask = result['seg_mask']
        all_probs = result['all_probs']
        model_type = MODEL_CONFIGS[model_key]['type']

        # ORIGINAL IMAGE
        ax1 = plt.subplot(n_models, 3, row * 3 + 1)
        ax1.imshow(img_original, cmap='gray')
        title_text = f'{title_prefix}{model_name}'
        ax1.set_title(title_text, fontsize=10, fontweight='bold', color=border_color, pad=8)
        ax1.axis('off')
        for spine in ax1.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4 if is_best else 1.5)
        
        # SEGMENTATION
        ax2 = plt.subplot(n_models, 3, row * 3 + 2)
        if model_type == 'classifier':
            ax2.text(0.5, 0.5, "No Mask Output\n(Classifier Only)", 
                    ha='center', va='center', fontsize=10, color='gray')
            ax2.set_title('Segmentation', fontsize=10, fontweight='bold', pad=8)
            ax2.axis('off')
        else:
            ax2.imshow(seg_mask, cmap='hot')
            tumor_pct = np.mean(seg_mask)
            if tumor_pct > 0.05:
                tumor_status = 'HIGH'
                status_color = 'red'
            elif tumor_pct > 0.01:
                tumor_status = 'MEDIUM'
                status_color = 'orange'
            else:
                tumor_status = 'LOW'
                status_color = 'blue'
            ax2.set_title(f'{tumor_status} Tumor: {tumor_pct:.2%}', 
                         fontsize=10, fontweight='bold', color=status_color, pad=8)
            ax2.axis('off')
        for spine in ax2.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4 if is_best else 1.5)
        
        # CLASSIFICATION
        ax3 = plt.subplot(n_models, 3, row * 3 + 3)
        if model_type in ['attention', 'unet']:
            ax3.text(0.5, 0.5, "No Class Output\n(Segmentation Only)", 
                    ha='center', va='center', fontsize=10, color='gray')
            ax3.set_title('Classification', fontsize=10, fontweight='bold', pad=8)
            ax3.axis('off')
        else:
            class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
            # ===== BLUE THEME =====
            colors = ['blue' if idx_to_class[i] == pred_class else 'lightgray' 
                     for i in sorted(idx_to_class.keys())]
            
            bars = ax3.barh(class_names, all_probs, color=colors, height=0.6, edgecolor='black', linewidth=1)
            ax3.set_xlim(0, 1.05)
            
            # ===== CONFIDENCE LEVEL =====
            if confidence > 0.8:
                conf_indicator = 'HIGH'
                title_color = 'blue'
            elif confidence > 0.6:
                conf_indicator = 'MEDIUM'
                title_color = 'orange'
            else:
                conf_indicator = 'LOW'
                title_color = 'red'

            ax3.set_title(f'[{conf_indicator}] {pred_class}: {confidence:.1%}', 
                         fontsize=10, fontweight='bold', color=title_color, pad=8)
            ax3.grid(axis='x', alpha=0.3, linestyle='--')
            ax3.tick_params(labelsize=9)
            ax3.set_xlabel('Probability', fontsize=8)
            
            for i, (bar, prob) in enumerate(zip(bars, all_probs)):
                if prob > 0.03:
                    ax3.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontsize=9, fontweight='bold')
        
        for spine in ax3.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4 if is_best else 1.5)
        
        row += 1

    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, hspace=0.4, wspace=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',pad_inches = 0)
        print(f"  Saved plot to {save_path}")
    
    plt.close()
    print(">>> AFTER VISUALIZE")

def compare_models_overlay(img_original, gt_mask, all_results, save_path=None):

    img_original = np.squeeze(img_original)
    gt_mask = np.squeeze(gt_mask)

    seg_results = {}

    for model_key, result in all_results.items():
        model_type = MODEL_CONFIGS[model_key]['type']

        if model_type != 'classifier':
            seg_results[model_key] = result

    n_models = len(seg_results)

    fig, axes = plt.subplots(
        1,
        n_models + 1,
        figsize=(5 * (n_models + 1), 5)
    )

    # ===== ORIGINAL =====
    axes[0].imshow(img_original, cmap='gray', vmin=0, vmax=255)

    axes[0].set_title(
        "Original MRI",
        fontsize=13,
        fontweight='bold'
    )

    axes[0].axis('off')

    idx = 1

    for model_key, result in seg_results.items():

        pred_mask = np.squeeze(result['seg_mask'])

        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        gt_bin = (gt_mask > 0.5).astype(np.uint8)

        model_name = MODEL_CONFIGS[model_key]['name']

        overlay = cv2.cvtColor(img_original.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        overlay[gt_bin == 1] = [0, 0, 255]

        overlay[pred_mask == 1] = [255, 0, 0]

        overlap = (gt_bin == 1) & (pred_mask == 1)
        overlay[overlap] = [255, 0, 255]

        axes[idx].imshow(overlay)

        tumor_pct = np.mean(pred_mask) * 100

        axes[idx].set_title(
            f"{model_name}\nTumor: {tumor_pct:.2f}%",
            fontsize=12,
            fontweight='bold'
        )

        axes[idx].axis('off')

        idx += 1

    fig.suptitle(
        "Ground Truth (Blue) vs Prediction (Red)",
        fontsize=16,
        fontweight='bold'
    )

    plt.tight_layout()

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight'
        )

        print(f"Overlay saved to: {save_path}")

    plt.close()