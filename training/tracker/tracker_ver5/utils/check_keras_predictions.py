import os
import sys
import numpy as np
import tensorflow as tf
import h5py

project_dir = "/home/elazarkin/work/projects/smart_rahfan/training/tracker/tracker_ver4"
model_path = os.path.join(project_dir, "outputs/tracker_coords_qat.keras")
h5_path = os.path.join(project_dir, "dataset_generator/compiled/dataset.h5")

sys.path.append(project_dir)
import tracker_model
import tensorflow_model_optimization as tfmot
from utils.quantization_optimization import CustomLayerQuantizeConfig

print(f"Loading QAT Keras model from {model_path}...")
custom_objects = {
    "DepthwiseCorrelationFusion": tracker_model.DepthwiseCorrelationFusion,
    "DepthToSpace": tracker_model.DepthToSpace,
    "CustomLayerQuantizeConfig": CustomLayerQuantizeConfig,
}

with tfmot.quantization.keras.quantize_scope(custom_objects):
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

print(f"\nLoading some samples from {h5_path}...")
with h5py.File(h5_path, 'r') as f:
    ref_stack = f['reference_stack'][:5].astype(np.float32)
    search_frame = f['search_frame'][:5].astype(np.float32)
    gt_coords = f['ground_truth_coords'][:5].astype(np.float32)

print("\nRunning QAT Keras inference on 5 samples...")
for i in range(5):
    hm, q = model([ref_stack[i:i+1], search_frame[i:i+1]], training=False)
    print(f"\nSample {i}:")
    print(f"  GT Coords: {gt_coords[i]}")
    print(f"  QAT Keras Heatmap max: {np.max(hm):.4f}")
    print(f"  QAT Keras Heatmap min: {np.min(hm):.4f}")
    print(f"  QAT Keras Quality: {q[0][0]:.4f}")
