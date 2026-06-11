#!/usr/bin/env python3
import os
import sys
import argparse
import h5py
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Prepare calibration dataset for RKNN conversion from HDF5 dataset")
    parser.add_argument(
        "--h5_path",
        default="../../training/tracker/tracker_ver4/dataset_generator/compiled/dataset.h5",
        help="Path to the training dataset.h5 file"
    )
    parser.add_argument(
        "--output_dir",
        default="configs/calibration_data",
        help="Directory where individual .npy calibration files will be saved"
    )
    parser.add_argument(
        "--dataset_txt",
        default="configs/dataset.txt",
        help="Path to the output RKNN dataset config text file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of calibration samples to extract (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # 1. Validate paths
    if not os.path.exists(args.h5_path):
        print(f"[ERROR] HDF5 dataset not found at: {args.h5_path}", file=sys.stderr)
        print("Please check the path or compile the dataset first.", file=sys.stderr)
        sys.exit(1)
        
    print(f"--> Loading dataset from: {args.h5_path}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    txt_dir = os.path.dirname(args.dataset_txt)
    if txt_dir:
        os.makedirs(txt_dir, exist_ok=True)
        
    # 2. Extract random samples
    with h5py.File(args.h5_path, 'r') as f:
        total_samples = f['reference_stack'].shape[0]
        print(f"Total samples available in HDF5: {total_samples}")
        
        num_samples = min(args.num_samples, total_samples)
        if num_samples < args.num_samples:
            print(f"[WARNING] Requested {args.num_samples} samples, but only {total_samples} are available. Capping to {total_samples}.")
            
        # Select random indices using seed for reproducibility
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(total_samples, size=num_samples, replace=False)
        
        print(f"--> Extracting {num_samples} random samples for calibration...")
        
        # We will write the list of files to the text file
        with open(args.dataset_txt, 'w', encoding='utf-8') as txt_file:
            for i, idx in enumerate(indices):
                # Extract input features
                ref_data = f['reference_stack'][idx]  # Shape: (1, 64, 64, 16)
                search_data = f['search_frame'][idx]  # Shape: (256, 256, 1)
                
                # Prepend batch dimension 1 since the ONNX model is built with static batch size = 1
                ref_expanded = np.expand_dims(ref_data, axis=0)      # Shape: (1, 1, 64, 64, 16)
                search_expanded = np.expand_dims(search_data, axis=0)  # Shape: (1, 256, 256, 1)
                
                # Define filenames
                ref_filename = os.path.join(args.output_dir, f"ref_{i}.npy")
                search_filename = os.path.join(args.output_dir, f"search_{i}.npy")
                
                # Save as numpy binaries
                np.save(ref_filename, ref_expanded.astype(np.float32))
                np.save(search_filename, search_expanded.astype(np.float32))
                
                # Write paths to dataset.txt (separated by space)
                txt_file.write(f"{ref_filename} {search_filename}\n")
                
                if (i + 1) % 100 == 0 or (i + 1) == num_samples:
                    print(f"  Processed {i + 1}/{num_samples} samples...")
                    
    print(f"[SUCCESS] Calibration dataset generated successfully!")
    print(f"  Individual .npy files saved in: {args.output_dir}")
    print(f"  RKNN dataset description file written to: {args.dataset_txt}")

if __name__ == "__main__":
    main()
