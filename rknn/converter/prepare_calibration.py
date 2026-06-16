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
        default="../../training/tracker/tracker_ver5/dataset_generator/compiled/dataset.h5",
        help="Path to the training dataset.h5 file"
    )
    parser.add_argument(
        "--keras_path",
        default="../../training/tracker/tracker_ver5/outputs/tracker_model_fbn.keras",
        help="Path to the trained Keras model file to run intermediate feature inference"
    )
    parser.add_argument(
        "--output_dir",
        default="configs/calibration_data",
        help="Directory where individual .npy calibration files will be saved"
    )
    parser.add_argument(
        "--dataset_template_txt",
        default="configs/dataset_template.txt",
        help="Path to the template model calibration descriptor text file"
    )
    parser.add_argument(
        "--dataset_frame_txt",
        default="configs/dataset_frame.txt",
        help="Path to the frame model calibration descriptor text file"
    )
    parser.add_argument(
        "--dataset_full_txt",
        default="configs/dataset_full.txt",
        help="Path to the full model calibration descriptor text file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of calibration samples to extract (default: 100)"
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
        
    if not os.path.exists(args.keras_path):
        print(f"[ERROR] Keras model file not found at: {args.keras_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"--> Loading dataset from: {args.h5_path}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    for txt_file in [args.dataset_template_txt, args.dataset_frame_txt, args.dataset_full_txt]:
        txt_dir = os.path.dirname(txt_file)
        if txt_dir:
            os.makedirs(txt_dir, exist_ok=True)
            
    # Load TensorFlow and Keras model to run intermediate feature inference
    print("[*] Loading TensorFlow and the Keras model...")
    try:
        import tensorflow as tf
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(script_dir, "../../training/tracker/tracker_ver5"))
        import tracker_model
        
        custom_objects = {
            "DepthwiseCorrelationFusion": tracker_model.DepthwiseCorrelationFusion,
            "Conv2DCorrelationFusion": tracker_model.Conv2DCorrelationFusion,
            "DepthToSpace": tracker_model.DepthToSpace,
            "HeatmapNormalization": tracker_model.HeatmapNormalization,
        }
        
        keras_model = tf.keras.models.load_model(args.keras_path, compile=False, custom_objects=custom_objects, safe_mode=False)
        ref_encoder = keras_model.get_layer("reference_target_encoder")
        print("[+] Model and Reference Target Encoder loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load Keras model for intermediate inference: {e}", file=sys.stderr)
        sys.exit(1)
        
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
        
        # Open descriptor files for writing
        with open(args.dataset_template_txt, 'w', encoding='utf-8') as f_template, \
             open(args.dataset_frame_txt, 'w', encoding='utf-8') as f_frame, \
             open(args.dataset_full_txt, 'w', encoding='utf-8') as f_full:
             
            for i, idx in enumerate(indices):
                # Extract input features from dataset
                ref_data = f['reference_stack'][idx]  # Shape: (64, 64, 16)
                search_data = f['search_frame'][idx]  # Shape: (256, 256, 1)
                
                # Prepend batch dimension 1
                ref_expanded = np.expand_dims(ref_data, axis=0)      # Shape: (1, 64, 64, 16)
                search_expanded = np.expand_dims(search_data, axis=0)  # Shape: (1, 256, 256, 1)
                
                # Run reference encoder to get intermediate NPU features
                ref_tensor = tf.cast(ref_expanded, tf.float32)
                # Normalize if not already normalized in H5
                if tf.reduce_max(ref_tensor) > 1.001:
                    ref_tensor = ref_tensor / 255.0
                features_tensor = ref_encoder(ref_tensor, training=False)
                features_data = features_tensor.numpy()  # Shape: (1, 8, 8, 64)
                
                # Transpose all arrays from NHWC to NCHW to match RKNN's internal NPU format
                ref_nchw = np.transpose(ref_expanded, (0, 3, 1, 2))      # Shape: (1, 16, 64, 64)
                search_nchw = np.transpose(search_expanded, (0, 3, 1, 2))  # Shape: (1, 1, 256, 256)
                features_nchw = np.transpose(features_data, (0, 3, 1, 2))  # Shape: (1, 64, 8, 8)
                
                # Define filenames
                template_filename = os.path.join(args.output_dir, f"template_{i}.npy")
                search_filename = os.path.join(args.output_dir, f"search_{i}.npy")
                features_filename = os.path.join(args.output_dir, f"features_{i}.npy")
                
                # Save as numpy binaries (float32 matches model's activation dtype before quantization)
                np.save(template_filename, ref_nchw.astype(np.float32))
                np.save(search_filename, search_nchw.astype(np.float32))
                np.save(features_filename, features_nchw.astype(np.float32))
                
                # Resolve paths relative to their descriptor text directories
                template_rel_temp = os.path.relpath(template_filename, os.path.dirname(args.dataset_template_txt))
                search_rel_frame = os.path.relpath(search_filename, os.path.dirname(args.dataset_frame_txt))
                features_rel_frame = os.path.relpath(features_filename, os.path.dirname(args.dataset_frame_txt))
                
                template_rel_full = os.path.relpath(template_filename, os.path.dirname(args.dataset_full_txt))
                search_rel_full = os.path.relpath(search_filename, os.path.dirname(args.dataset_full_txt))
                
                # Write paths to descriptor files
                f_template.write(f"{template_rel_temp}\n")
                f_frame.write(f"{search_rel_frame} {features_rel_frame}\n")
                f_full.write(f"{template_rel_full} {search_rel_full}\n")
                
                if (i + 1) % 10 == 0 or (i + 1) == num_samples:
                    print(f"  Processed {i + 1}/{num_samples} samples...")
                    
    print(f"[SUCCESS] Multi-Model Calibration dataset generated successfully!")
    print(f"  Individual .npy files saved in: {args.output_dir}")
    print(f"  - Template descriptor: {args.dataset_template_txt}")
    print(f"  - Frame descriptor:    {args.dataset_frame_txt}")
    print(f"  - Full descriptor:     {args.dataset_full_txt}")

if __name__ == "__main__":
    main()
