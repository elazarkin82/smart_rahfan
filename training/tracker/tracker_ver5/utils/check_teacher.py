import os
import sys
import numpy as np
import tensorflow as tf
import h5py

project_dir = "/home/elazarkin/work/projects/smart_rahfan/training/tracker/tracker_ver4"
model_path = os.path.join(project_dir, "outputs/tracker_coords_fbn.keras")
h5_path = os.path.join(project_dir, "dataset_generator/compiled/dataset.h5")

sys.path.append(project_dir)
import tracker_model

print(f"Loading teacher model from {model_path}...")
custom_objects = {
    "DepthwiseCorrelationFusion": tracker_model.DepthwiseCorrelationFusion,
    "DepthToSpace": tracker_model.DepthToSpace,
}

model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False, custom_objects=custom_objects)

print(f"\nLoading some samples from {h5_path}...")
with h5py.File(h5_path, 'r') as f:
    ref_stack = f['reference_stack'][:5].astype(np.float32)
    search_frame = f['search_frame'][:5].astype(np.float32)
    gt_heatmap = f['ground_truth_heatmap'][:5].astype(np.float32)

print("\nRunning teacher inference on 5 samples...")
for i in range(5):
    hm, q = model([ref_stack[i:i+1], search_frame[i:i+1]], training=False)
    print(f"\nSample {i}:")
    print(f"  GT Heatmap max: {np.max(gt_heatmap[i]):.4f}")
    print(f"  Teacher Heatmap max: {np.max(hm):.4f}")
    print(f"  Teacher Heatmap min: {np.min(hm):.4f}")
    print(f"  Teacher Heatmap mean: {np.mean(hm):.4f}")
    print(f"  Teacher Quality: {q[0][0]:.4f}")
