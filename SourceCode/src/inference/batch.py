from configs.inference_config import MODEL_CONFIGS
from src.inference.pipeline import run_all_models_inference
from pathlib import Path

def batch_inference(image_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(image_folder).glob(ext))
    
    image_paths = list(set([p.resolve() for p in image_paths]))
    image_paths = sorted(image_paths)
    
    if not image_paths:
        print(f"No images found in {image_folder}")
        return

    print(f"Found {len(image_paths)} images. Processing...\n")
    
    success_count = 0
    fail_count = 0
    
    for idx, img_path in enumerate(image_paths, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(image_paths)}] Processing: {img_path.name}")
        print(f"{'='*70}")
        
        try:
            run_all_models_inference(img_path, output_folder)
            success_count += 1
            print(f"✓ SUCCESS: {img_path.name}")
        except Exception as e:
            fail_count += 1
            print(f"\nFAILED: {img_path.name}")
            print(f"Error: {e}")
            print("\nFull traceback:")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"Batch processing complete!")
    print(f"Success: {success_count}/{len(image_paths)}")
    print(f"Failed: {fail_count}/{len(image_paths)}")
    print(f"Results in {output_folder}")
    print(f"{'='*70}")
