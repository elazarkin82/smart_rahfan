import os
import h5py
import numpy as np
import tensorflow as tf

MODEL_PATH = "/home/elazarkin/work/projects/smart_rahfan/training/tracker/tracker_ver5/outputs/tracker_matmul_corr_with_quality_fbn.keras"
#MODEL_PATH = "/home/elazarkin/work/projects/smart_rahfan/training/tracker/tracker_ver5/outputs/tracker_matmul_corr_fbn.keras"
DATASET_PATH = "/home/elazarkin/work/projects/smart_rahfan/training/tracker/tracker_ver5/dataset_generator/compiled/dataset.h5"

def main():
    print(f"[*] Loading Keras model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # 1. Convert to TFLite (Float32)
    print("[*] Converting Keras model to unquantized TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    
    tflite_path = "matmul_corr.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[+] Saved TFLite model to {tflite_path}")

    # 2. Extract dataset sample and run Keras prediction
    print(f"[*] Loading sample from dataset: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
        
    with h5py.File(DATASET_PATH, "r") as f:
        # Load the very first sample
        ref_stack = f["reference_stack"][0:1]    # shape (1, 128, 128, 2)
        search_frame = f["search_frame"][0:1]    # shape (1, 256, 256, 1)

    print("[*] Running Keras inference...")
    pred_heatmap, pred_quality = model.predict(
        {"reference_stack": ref_stack, "search_frame": search_frame},
        verbose=0
    )
    
    # Save inputs to binary files (float32, row-major flat layout)
    print("[*] Writing inputs and outputs to flat binary files...")
    (ref_stack * 255.0).astype(np.float32).tofile("ref_stack.bin")
    (search_frame * 255.0).astype(np.float32).tofile("search_frame.bin")
    
    # Save reference outputs (removing the batch dimension)
    pred_heatmap[0].astype(np.float32).tofile("keras_heatmap.bin")
    pred_quality[0].astype(np.float32).tofile("keras_quality.bin")
    
    print("[+] Flat binary files created successfully:")
    print(f"  - ref_stack.bin (shape: {list(ref_stack.shape[1:])}, float32)")
    print(f"  - search_frame.bin (shape: {list(search_frame.shape[1:])}, float32)")
    print(f"  - keras_heatmap.bin (shape: {list(pred_heatmap.shape[1:])}, float32)")
    print(f"  - keras_quality.bin (shape: {list(pred_quality.shape[1:])}, float32)")

if __name__ == "__main__":
    main()
