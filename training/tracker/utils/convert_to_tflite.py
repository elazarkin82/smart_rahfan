#!/usr/bin/env python3
"""
Keras to TensorFlow Lite (TFLite) Converter Script
==================================================
This utility loads a trained Keras model (such as TargetTracker2) and exports
it as a highly-optimized TFLite model suitable for mobile and edge deployments.

Usage:
    python3 convert_to_tflite.py --keras_in <path_to_keras_model> --tflite_out <path_to_tflite_output>
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert Keras models (.keras) to TensorFlow Lite (.tflite)")
    parser.add_argument(
        "--keras_in",
        required=True,
        help="Path to the input Keras model file (.keras)"
    )
    parser.add_argument(
        "--tflite_out",
        required=True,
        help="Path to save the converted TFLite model (.tflite)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.keras_in):
        print(f"Error: Input Keras model file '{args.keras_in}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    # Ensure parent directory for output exists
    output_dir = os.path.dirname(args.tflite_out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"[*] Importing TensorFlow...")
    try:
        import tensorflow as tf
    except ImportError:
        print("Error: TensorFlow is not installed. Please install it with: pip install tensorflow", file=sys.stderr)
        sys.exit(1)
        
    print(f"[*] Loading Keras model from: {args.keras_in} ...")
    try:
        # Load with compile=False to bypass custom loss function requirements during inference conversion
        model = tf.keras.models.load_model(args.keras_in, compile=False)
        print("[+] Model loaded successfully.")
    except Exception as e:
        print(f"Error: Failed to load the Keras model. Details:\n{e}", file=sys.stderr)
        sys.exit(1)
        
    print("[*] Initializing TFLite Converter...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Force a pure unquantized FLOAT32 model conversion.
        # This completely avoids TFLite runtime errors with TRANSPOSE_CONV layers
        # (where hybrid INT8 weights and FLOAT32 inputs are not supported on CPU).
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        print("[*] Converting model to TFLite format (this may take a minute)...")
        tflite_model = converter.convert()
        print("[+] Model conversion completed.")
    except Exception as e:
        print(f"Error: Model conversion failed. Details:\n{e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"[*] Saving TFLite model to: {args.tflite_out} ...")
    try:
        with open(args.tflite_out, "wb") as f:
            f.write(tflite_model)
        print(f"[SUCCESS] TFLite model saved successfully at: {args.tflite_out}")
    except Exception as e:
        print(f"Error: Failed to save the TFLite model. Details:\n{e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
