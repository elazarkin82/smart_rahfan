import os
import glob
import json
import pickle
import numpy as np
import cv2
import tqdm

def load_config(path="pipeline_config.json"):
    with open(path, 'r') as f:
        return json.load(f)

def generate_heatmap(shape, center, sigma):
    """Generates a 2D Gaussian heatmap centered at the specified target."""
    h, w = shape[:2]
    x = np.arange(0, w, 1, float)
    y = np.arange(0, h, 1, float)[:, np.newaxis]
    x0, y0 = center
    
    # 2D Gaussian formula
    heatmap = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return np.expand_dims(heatmap, axis=-1)

def get_crop(image, center_x, center_y, size):
    """Extracts a square crop of 'size' around the center, padding with edge replication if out of bounds."""
    h, w = image.shape[:2]
    half = size / 2.0
    
    x1 = int(round(center_x - half))
    x2 = int(round(center_x + half))
    y1 = int(round(center_y - half))
    y2 = int(round(center_y + half))
    
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Extract
    crop = image[y1:y2, x1:x2].copy()
    
    # Pad if we hit the edge of the screen
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
        
    return crop

def build_reference_stack(image, center, num_layers, max_size, min_size, target_size):
    """
    Builds the Multi-Scale Reference Stack.
    Creates 'num_layers' crops ranging from 'max_size' down to 'min_size'.
    All crops are resized to 'target_size'x'target_size' and stacked into a single tensor.
    """
    sizes = np.linspace(max_size, min_size, num_layers)
    stack_layers = []
    
    for sz in sizes:
        crop = get_crop(image, center[0], center[1], sz)
        # Resize to uniform shape (e.g., 16x16)
        resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        stack_layers.append(resized)
        
    # Stack along first axis to form (Layers, H, W, C)
    # Output dtype uint8 to save space in the PKL
    return np.stack(stack_layers, axis=0).astype(np.uint8)

def main():
    config = load_config()
    cache_dir = config['generation'].get('cache_dir', 'cache')
    compiler_cfg = config['compiler']
    
    dataset_dir = compiler_cfg['dataset_dir']
    os.makedirs(dataset_dir, exist_ok=True)
    
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "flight_*.pkl")))
    
    if not cache_files:
        print(f"No cache files found in '{cache_dir}'. Run dataset_generator.py first.")
        return
        
    print(f"Found {len(cache_files)} cached flights. Starting compilation...")
    
    # Compiler settings
    layers = compiler_cfg['stack_layers']
    max_sz = compiler_cfg['crop_max_size']
    min_sz = compiler_cfg['crop_min_size']
    tgt_sz = compiler_cfg['stack_target_size']
    sigma = compiler_cfg['heatmap_sigma']
    
    processed_count = 0
    
    for cache_path in tqdm.tqdm(cache_files, desc="Compiling Dataset"):
        basename = os.path.basename(cache_path)
        out_name = basename.replace("flight_", "train_")
        out_path = os.path.join(dataset_dir, out_name)
        
        # Skip if already compiled
        if os.path.exists(out_path):
            continue
            
        with open(cache_path, 'rb') as f:
            flight_data = pickle.load(f)
            
        if not flight_data or len(flight_data) < 2:
            continue
            
        # 1. Reference Initialization (from frame 0)
        frame_0 = flight_data[0]
        ref_stack = build_reference_stack(
            frame_0['image_rgb'], 
            frame_0['target_2d'], 
            layers, max_sz, min_sz, tgt_sz
        )
        
        # 2. Training Pairs (frames 1..N)
        training_samples = []
        for frame_dict in flight_data[1:]:
            search_frame = frame_dict['image_rgb']
            target_2d = frame_dict['target_2d']
            
            heatmap = generate_heatmap(search_frame.shape, target_2d, sigma)
            
            sample = {
                "reference_stack": ref_stack,               # Shape: (16, 16, 16, 3)
                "search_frame": search_frame,               # Shape: (H, W, 3)
                "ground_truth_heatmap": heatmap.astype(np.float16), # Shape: (H, W, 1) Float16 to save space
                "metadata": {
                    "flight_id": basename,
                    "frame_idx": frame_dict['frame_index'],
                    "target_2d": target_2d,
                    "distance": frame_dict['distance_to_target']
                }
            }
            training_samples.append(sample)
            
        # 3. Save to dataset dir
        with open(out_path, 'wb') as f:
            pickle.dump(training_samples, f)
            
        processed_count += 1
        
    print(f"\nCompilation finished! {processed_count} new flights compiled into '{dataset_dir}'.")
    print(f"Dataset is ready for Neural Network training.")

if __name__ == '__main__':
    main()
