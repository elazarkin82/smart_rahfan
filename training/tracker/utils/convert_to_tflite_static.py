#!/usr/bin/env python3
"""
TargetTrackerVer4 Static TFLite Converter
=========================================
This script reconstructs the Siamese-Attention architecture of Tracker Ver 4,
loads only the trained weights from a Keras model (.keras), and exports it
as a highly-optimized static TFLite model (batch size = 1).

This completely avoids:
  1. Keras version mismatch errors (Keras 3 vs Keras 2 'tf_keras' namespace errors).
  2. Python version marshal/bytecode compatibility errors with Lambda layers.
  3. TFLite dynamic shape and BROADCAST_TO runtime exceptions on Android.

Usage:
    python3 convert_to_tflite_static.py --keras_in outputs/tracker.keras --tflite_out outputs/tracker.tflite --copy_to_android
"""

import os
import sys
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser(description="Statically convert Tracker Ver 4 Keras model to TFLite.")
    parser.add_argument(
        "--keras_in",
        default="outputs/tracker.keras",
        help="Path to input Keras model weights/file (.keras)"
    )
    parser.add_argument(
        "--tflite_out",
        default="outputs/tracker.tflite",
        help="Path to save the compiled static TFLite model (.tflite)"
    )
    parser.add_argument(
        "--copy_to_android",
        action="store_true",
        help="Copy the compiled TFLite model directly to Android assets"
    )
    
    args = parser.parse_args()
    
    # 1. Validate inputs
    if not os.path.exists(args.keras_in):
        print(f"Error: Keras model file '{args.keras_in}' not found.", file=sys.stderr)
        sys.exit(1)
        
    print("[*] Importing TensorFlow...")
    try:
        import tensorflow as tf
    except ImportError:
        print("Error: TensorFlow is not installed in this Python environment.", file=sys.stderr)
        sys.exit(1)
        
    # Append current dir to sys.path to load tracker_model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    
    try:
        from tracker_model import TargetTrackerVer4
    except ImportError:
        print("Error: Could not import TargetTrackerVer4 from tracker_model.py. Ensure this script is run from the tracker_ver4 folder.", file=sys.stderr)
        sys.exit(1)
        
    # 2. Reconstruct Model Architecture
    print("[*] Reconstructing TargetTrackerVer4 model architecture...")
    try:
        tracker = TargetTrackerVer4()
        model = tracker.create_model()
        print("[+] Model architecture successfully built.")
    except Exception as e:
        print(f"Error: Failed to build model architecture. Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 3. Load weights only (bypasses any serialized bytecode/namespace issues)
    print(f"[*] Loading trained weights from: {args.keras_in} ...")
    try:
        model.load_weights(args.keras_in)
        print("[+] Weights loaded successfully.")
    except Exception as e:
        print(f"Error: Failed to load weights. Ensure the Keras file contains compatible weights.\nDetails: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 4. Generate Concrete Function with Static Batch Shape (batch_size = 1)
    print("[*] Defining concrete function with static input shapes (forcing batch_size=1)...")
    try:
        @tf.function
        def run_model(ref, search):
            return model([ref, search])
            
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec([1, 16, 32, 32, 1], tf.float32, name="reference_stack"),
            tf.TensorSpec([1, 256, 256, 1], tf.float32, name="search_frame")
        )
        print("[+] Static concrete function generated.")
    except Exception as e:
        print(f"Error: Failed to generate concrete function. Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 5. Convert to TFLite
    print("[*] Converting model to static TFLite format (this may take a moment)...")
    try:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        tflite_model = converter.convert()
        print("[+] Model converted to TFLite successfully.")
    except Exception as e:
        print(f"Error: TFLite conversion failed. Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 6. Save compiled model
    out_dir = os.path.dirname(args.tflite_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    try:
        with open(args.tflite_out, "wb") as f:
            f.write(tflite_model)
        print(f"[SUCCESS] Static TFLite model saved at: {args.tflite_out}")
    except Exception as e:
        print(f"Error: Failed to save TFLite model to '{args.tflite_out}'. Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 7. Optionally copy to Android assets
    if args.copy_to_android:
        android_assets_dir = os.path.join(script_dir, "android", "app", "src", "main", "assets")
        if os.path.exists(android_assets_dir):
            android_tflite_path = os.path.join(android_assets_dir, "tracker.tflite")
            try:
                shutil.copy2(args.tflite_out, android_tflite_path)
                print(f"[SUCCESS] Copied compiled model directly to Android assets: {android_tflite_path}")
            except Exception as e:
                print(f"Warning: Failed to copy model to Android assets. Details: {e}", file=sys.stderr)
        else:
            print(f"Warning: Android assets folder not found at: {android_assets_dir}. Skipping copy.", file=sys.stderr)

if __name__ == "__main__":
    main()
