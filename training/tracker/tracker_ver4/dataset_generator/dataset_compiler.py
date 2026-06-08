import os
import glob
import json
import pickle
import numpy as np
import cv2
import tqdm
import h5py

def load_config(path="pipeline_config.json"):
    if not os.path.exists(path) and path == "pipeline_config.json":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path)
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
        # Resize to uniform shape (e.g., 64x64)
        resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        stack_layers.append(resized)
        
    # Stack along channels axis to form (H, W, Layers)
    stack = np.stack(stack_layers, axis=-1).astype(np.uint8)
    # Expand dims on first axis to form (1, H, W, Layers)
    return np.expand_dims(stack, axis=0)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config()
    cache_dir = config['generation'].get('cache_dir', 'cache')
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(script_dir, cache_dir)
        
    compiler_cfg = config['compiler']
    dataset_dir = compiler_cfg['dataset_dir']
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(script_dir, dataset_dir)
        
    os.makedirs(dataset_dir, exist_ok=True)
    
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "flight_*.pkl")))
    
    if not cache_files:
        print(f"No cache files found in '{cache_dir}'. Run dataset_generator.py first.")
        return
        
    print(f"Found {len(cache_files)} cached flights. Starting compilation to HDF5...")
    
    # Compiler settings
    layers = compiler_cfg['stack_layers']
    max_sz = compiler_cfg['crop_max_size']
    min_sz = compiler_cfg['crop_min_size']
    tgt_sz = compiler_cfg['stack_target_size']
    sigma = compiler_cfg['heatmap_sigma']
    relative_sigma = compiler_cfg.get('heatmap_relative_sigma', None)
    neg_sample_ratio = compiler_cfg.get('synthetic_negative_ratio', compiler_cfg.get('negative_sample_ratio', 0.20))
    
    h5_path = os.path.join(dataset_dir, "dataset.h5")
    if os.path.exists(h5_path):
        try:
            os.remove(h5_path)
        except Exception as e:
            print(f"Warning: Could not remove existing dataset.h5: {e}")
            
    f_h5 = h5py.File(h5_path, 'w')
    
    # Extendable datasets
    ref_ds = f_h5.create_dataset('reference_stack', shape=(0, 1, tgt_sz, tgt_sz, layers), maxshape=(None, 1, tgt_sz, tgt_sz, layers), dtype='float32', chunks=(16, 1, tgt_sz, tgt_sz, layers))
    search_ds = f_h5.create_dataset('search_frame', shape=(0, 256, 256, 1), maxshape=(None, 256, 256, 1), dtype='float32', chunks=(16, 256, 256, 1))
    heatmap_ds = f_h5.create_dataset('ground_truth_heatmap', shape=(0, 256, 256, 1), maxshape=(None, 256, 256, 1), dtype='float32', chunks=(16, 256, 256, 1))
    quality_ds = f_h5.create_dataset('ground_truth_quality', shape=(0, 1), maxshape=(None, 1), dtype='float32', chunks=(16, 1))
    
    processed_count = 0
    
    for cache_path in tqdm.tqdm(cache_files, desc="Compiling Dataset"):
        basename = os.path.basename(cache_path)
        
        with open(cache_path, 'rb') as f:
            try:
                flight_data = pickle.load(f)
            except Exception as e:
                print(f"\nWarning: Failed to load {cache_path}: {e}")
                continue
            
        if not flight_data or len(flight_data) < 2:
            continue
            
        # 1. Reference Initialization (from frame 0)
        frame_0 = flight_data[0]
        flight_max_sz = frame_0.get("crop_max_size", max_sz)
        flight_min_sz = frame_0.get("crop_min_size", min_sz)
        ref_stack = build_reference_stack(
            frame_0['image_gray'], 
            frame_0['target_2d'], 
            layers, flight_max_sz, flight_min_sz, tgt_sz
        )
        ref_float = ref_stack.astype(np.float32) / 255.0
        
        # 2. Training Pairs (frames 1..N)
        flight_refs = []
        flight_searches = []
        flight_heatmaps = []
        flight_qualities = []
        
        idx = cache_files.index(cache_path)
        neg_flight_data = None
        neg_cache_path = None
        if len(cache_files) > 1:
            neg_idx = (idx + 1) % len(cache_files)
            neg_cache_path = cache_files[neg_idx]
            with open(neg_cache_path, 'rb') as f:
                neg_flight_data = pickle.load(f)
                
        for k, frame_dict in enumerate(flight_data[1:]):
            search_frame_gray = frame_dict['image_gray']
            target_2d = frame_dict['target_2d']
            
            # Determine square crop size (minimum image dimension)
            h_s, w_s = search_frame_gray.shape[:2]
            s_crop = min(h_s, w_s)
            half = s_crop / 2.0
            
            if target_2d is None:
                # Natural negative frame
                search_crop_gray = get_crop(search_frame_gray, w_s / 2.0, h_s / 2.0, s_crop)
                search_resized = cv2.resize(search_crop_gray, (256, 256), interpolation=cv2.INTER_LINEAR)
                search_float = search_resized.astype(np.float32) / 255.0
                search_float = np.expand_dims(search_float, axis=-1)
                
                heatmap_float = np.zeros((256, 256, 1), dtype=np.float32)
                
                flight_refs.append(ref_float[0])
                flight_searches.append(search_float)
                flight_heatmaps.append(heatmap_float)
                flight_qualities.append(np.array([0.0], dtype=np.float32))
                continue
                
            # Crop Search Frame with central 1/8 padding jittering
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, s_crop * 0.375)
            dx = distance * np.cos(angle)
            dy = distance * np.sin(angle)
            
            search_crop_gray = get_crop(search_frame_gray, target_2d[0] + dx, target_2d[1] + dy, s_crop)
            search_resized = cv2.resize(search_crop_gray, (256, 256), interpolation=cv2.INTER_LINEAR)
            search_float = search_resized.astype(np.float32) / 255.0
            search_float = np.expand_dims(search_float, axis=-1)
            
            local_target_2d = (half - dx, half - dy)
            
            # Scale target local coordinates to 256x256 space for correct heatmap scaling
            scale_factor = 256.0 / s_crop
            local_target_2d_scaled = (local_target_2d[0] * scale_factor, local_target_2d[1] * scale_factor)
            
            # Dynamic Isotropic Gaussian Heatmap on 256x256 space
            if relative_sigma is not None:
                sample_sigma = 256.0 * relative_sigma
            else:
                sample_sigma = sigma * scale_factor
                
            # Generate heatmap directly on 256x256 space
            heatmap_resized = generate_heatmap((256, 256, 1), local_target_2d_scaled, sample_sigma)
            heatmap_float = heatmap_resized.astype(np.float32)
            
            quality_score = 1.0
            
            flight_refs.append(ref_float[0])
            flight_searches.append(search_float)
            flight_heatmaps.append(heatmap_float)
            flight_qualities.append(np.array([quality_score], dtype=np.float32))
            
            # 2b. Negative Pair: target from flight i is NOT present in flight j
            if neg_flight_data is not None and len(neg_flight_data) > 1 and np.random.uniform(0, 1) < neg_sample_ratio:
                neg_k = (k % (len(neg_flight_data) - 1)) + 1
                neg_frame_dict = neg_flight_data[neg_k]
                
                search_frame_gray_neg = neg_frame_dict['image_gray']
                h_n, w_n = search_frame_gray_neg.shape[:2]
                s_crop_neg = min(h_n, w_n)
                
                search_crop_gray_neg = get_crop(search_frame_gray_neg, w_n / 2.0, h_n / 2.0, s_crop_neg)
                search_resized_neg = cv2.resize(search_crop_gray_neg, (256, 256), interpolation=cv2.INTER_LINEAR)
                search_float_neg = search_resized_neg.astype(np.float32) / 255.0
                search_float_neg = np.expand_dims(search_float_neg, axis=-1)
                
                heatmap_float_neg = np.zeros((256, 256, 1), dtype=np.float32)
                
                flight_refs.append(ref_float[0])
                flight_searches.append(search_float_neg)
                flight_heatmaps.append(heatmap_float_neg)
                flight_qualities.append(np.array([0.0], dtype=np.float32))
                
        # Append this flight's samples to the HDF5 datasets
        if flight_refs:
            n_existing = ref_ds.shape[0]
            n_new = len(flight_refs)
            
            ref_ds.resize(n_existing + n_new, axis=0)
            search_ds.resize(n_existing + n_new, axis=0)
            heatmap_ds.resize(n_existing + n_new, axis=0)
            quality_ds.resize(n_existing + n_new, axis=0)
            
            # reference_stack input requires shape (N, 1, 64, 64, 16)
            ref_ds[n_existing:] = np.expand_dims(np.stack(flight_refs, axis=0), axis=1)
            search_ds[n_existing:] = np.stack(flight_searches, axis=0)
            heatmap_ds[n_existing:] = np.stack(flight_heatmaps, axis=0)
            quality_ds[n_existing:] = np.stack(flight_qualities, axis=0)
            
        processed_count += 1
        
    f_h5.close()
    print(f"\nCompilation finished! {processed_count} flights compiled into a single HDF5 dataset: '{h5_path}'.")
    print(f"Dataset is ready for Neural Network training.")

if __name__ == '__main__':
    main()
