import os
import sys
import numpy as np
import tensorflow as tf
import h5py

project_dir = "/home/elazarkin/work/projects/smart_rahfan/training/tracker/tracker_ver4"
tflite_path = os.path.join(project_dir, "outputs/tracker_qat.tflite")
h5_path = os.path.join(project_dir, "dataset_generator/compiled/dataset.h5")

print(f"Loading TFLite model from {tflite_path}...")
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n--- INPUT TENSORS ---")
for detail in input_details:
    print(f"Name: {detail['name']}")
    print(f"  Shape: {detail['shape']}")
    print(f"  Type: {detail['dtype']}")
    print(f"  Quantization: {detail['quantization']}")

print("\n--- OUTPUT TENSORS ---")
for detail in output_details:
    print(f"Name: {detail['name']}")
    print(f"  Shape: {detail['shape']}")
    print(f"  Type: {detail['dtype']}")
    print(f"  Quantization: {detail['quantization']}")

# Map input/output details
ref_idx = None
search_idx = None
for detail in input_details:
    if "reference_stack" in detail['name'].lower():
        ref_idx = detail['index']
    elif "search_frame" in detail['name'].lower():
        search_idx = detail['index']

if ref_idx is None or search_idx is None:
    # Fallback by shape
    for detail in input_details:
        if 16 in detail['shape'] or 64 in detail['shape']:
            ref_idx = detail['index']
        else:
            search_idx = detail['index']

heatmap_idx = None
quality_idx = None
for detail in output_details:
    if 256 in detail['shape']:
        heatmap_idx = detail['index']
    else:
        quality_idx = detail['index']

print(f"\nMapped inputs: ref={ref_idx}, search={search_idx}")
print(f"Mapped outputs: heatmap={heatmap_idx}, quality={quality_idx}")

# Run inference on some samples
print(f"\nLoading some samples from {h5_path}...")
with h5py.File(h5_path, 'r') as f:
    ref_stack = f['reference_stack'][:5].astype(np.float32)
    search_frame = f['search_frame'][:5].astype(np.float32)
    gt_heatmap = f['ground_truth_heatmap'][:5].astype(np.float32)

print("Running inference on 5 samples...")
for i in range(5):
    interpreter.set_tensor(ref_idx, ref_stack[i:i+1])
    interpreter.set_tensor(search_idx, search_frame[i:i+1])
    interpreter.invoke()
    
    pred_hm = interpreter.get_tensor(heatmap_idx)
    pred_q = interpreter.get_tensor(quality_idx)
    
    print(f"\nSample {i}:")
    print(f"  GT Heatmap max: {np.max(gt_heatmap[i]):.4f}")
    print(f"  Pred Heatmap max: {np.max(pred_hm):.4f}")
    print(f"  Pred Heatmap min: {np.min(pred_hm):.4f}")
    print(f"  Pred Heatmap mean: {np.mean(pred_hm):.4f}")
    print(f"  Pred Quality: {pred_q[0][0]:.4f}")
