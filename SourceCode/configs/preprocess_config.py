from pathlib import Path

DATA_DIR = Path('data/raw/brisc2025')
PROCESSED_DIR = Path('data/processed')
CONFIG = {
    'target_size': 224,
    'normalize': True,
    'apply_clahe': True,
    'clahe_clip_limit': 4.0,
    'clahe_tile_size': (8, 8),
    'denoise': True,
    'adaptive_threshold_masks': False,
}