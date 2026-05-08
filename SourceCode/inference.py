import argparse
import sys
from src.inference.batch import batch_inference
from src.inference.pipeline import run_all_models_inference
def main():
    parser = argparse.ArgumentParser(description='Universal Brain Tumor Inference - All Models')
    
    parser.add_argument('--image', type=str, help='Path to a single image file')
    parser.add_argument('--folder', type=str, help='Path to a folder of images')
    parser.add_argument('--output', type=str, default='results/demo_results', 
                        help='Folder to save results')
    
    args = parser.parse_args()
    
    if args.image:
        run_all_models_inference(args.image, args.output)
    elif args.folder:
        batch_inference(args.folder, args.output)
        
if __name__ == '__main__':
    main()