#!/usr/bin/env python3
"""
Quantization-Aware Training (QAT) Optimization Script
======================================================
This script applies QAT to a pre-trained Keras model to prepare it for 
high-accuracy INT8 TFLite conversion. It supports:
  1. Selective QAT wrapping to bypass custom/subclassed layers.
  2. Teacher-Student Knowledge Distillation mode.
  3. Direct dataset label training mode.
"""

import os
import sys
import argparse

import tensorflow as tf
# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass

# Run in graph mode (run_functions_eagerly=False) which is required to update MovingAverageQuantizer.
tf.config.run_functions_eagerly(False)

import tensorflow_model_optimization as tfmot

import numpy as np
import tqdm

class MemorySafeDataset:
    def __init__(self, h5_path, batch_size=16):
        import h5py
        import psutil
        import numpy as np
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 dataset not found at {h5_path}")
            
        self.h5_path = h5_path
        self.batch_size = batch_size
        
        with h5py.File(h5_path, 'r') as f:
            self.total_samples = f['reference_stack'].shape[0]
            
        # Calculate dataset size in bytes
        sample_size_bytes = (64 * 64 * 16 + 256 * 256 * 1 + 256 * 256 * 1) * 4 + 4
        total_expected_bytes = self.total_samples * sample_size_bytes
        
        mem = psutil.virtual_memory()
        
        # If the dataset fits in 35% of available memory, preload it into RAM once to avoid background thread leaks
        if total_expected_bytes < mem.available * 0.35:
            print(f"[*] Pre-loading dataset into RAM ({self.total_samples} samples, {total_expected_bytes / (1024**3):.2f} GB)...")
            with h5py.File(h5_path, 'r') as f:
                self.ref_all = f['reference_stack'][:]
                self.search_all = f['search_frame'][:]
                self.gt_hm_all = f['ground_truth_heatmap'][:]
                self.gt_q_all = f['ground_truth_quality'][:]
            self.preload_mode = True
        else:
            print(f"[*] Dataset size ({total_expected_bytes / (1024**3):.2f} GB) exceeds safe RAM limits. Streaming from HDF5...")
            self.preload_mode = False

    def get_generator(self):
        import numpy as np
        import h5py
        
        total_samples = self.total_samples
        batch_size = self.batch_size
        
        if self.preload_mode:
            def generator():
                indices = np.arange(total_samples)
                np.random.RandomState(42).shuffle(indices)
                
                num_batches = total_samples // batch_size
                for idx in range(num_batches):
                    batch_idx = indices[idx * batch_size : (idx + 1) * batch_size]
                    
                    ref = self.ref_all[batch_idx].astype(np.float32)
                    search = self.search_all[batch_idx].astype(np.float32)
                    gt_hm = self.gt_hm_all[batch_idx].astype(np.float32)
                    gt_q = self.gt_q_all[batch_idx].astype(np.float32)
                    
                    ref_tensor = tf.convert_to_tensor(ref, dtype=tf.float32)
                    search_tensor = tf.convert_to_tensor(search, dtype=tf.float32)
                    gt_hm_tensor = tf.convert_to_tensor(gt_hm, dtype=tf.float32)
                    gt_q_tensor = tf.convert_to_tensor(gt_q, dtype=tf.float32)
                    
                    inputs = {"reference_stack": ref_tensor, "search_frame": search_tensor}
                    targets = {"predicted_heatmap": gt_hm_tensor, "predicted_quality": gt_q_tensor}
                    
                    yield inputs, targets
                    
                    del ref, search, gt_hm, gt_q
                    del ref_tensor, search_tensor, gt_hm_tensor, gt_q_tensor
                    del inputs, targets
            return generator()
        else:
            def generator():
                indices = np.arange(total_samples)
                np.random.RandomState(42).shuffle(indices)
                
                num_batches = total_samples // batch_size
                with h5py.File(self.h5_path, 'r') as f:
                    for idx in range(num_batches):
                        batch_idx = indices[idx * batch_size : (idx + 1) * batch_size]
                        
                        # Sort indices for fast HDF5 slicing
                        sorted_order = np.argsort(batch_idx)
                        sorted_batch_idx = batch_idx[sorted_order]
                        
                        ref = f['reference_stack'][sorted_batch_idx].astype(np.float32)
                        search = f['search_frame'][sorted_batch_idx].astype(np.float32)
                        gt_hm = f['ground_truth_heatmap'][sorted_batch_idx].astype(np.float32)
                        gt_q = f['ground_truth_quality'][sorted_batch_idx].astype(np.float32)
                        
                        # Restore original shuffled order
                        rev_order = np.argsort(sorted_order)
                        ref = ref[rev_order]
                        search = search[rev_order]
                        gt_hm = gt_hm[rev_order]
                        gt_q = gt_q[rev_order]
                        
                        ref_tensor = tf.convert_to_tensor(ref, dtype=tf.float32)
                        search_tensor = tf.convert_to_tensor(search, dtype=tf.float32)
                        gt_hm_tensor = tf.convert_to_tensor(gt_hm, dtype=tf.float32)
                        gt_q_tensor = tf.convert_to_tensor(gt_q, dtype=tf.float32)
                        
                        inputs = {"reference_stack": ref_tensor, "search_frame": search_tensor}
                        targets = {"predicted_heatmap": gt_hm_tensor, "predicted_quality": gt_q_tensor}
                        
                        yield inputs, targets
                        
                        del ref, search, gt_hm, gt_q
                        del ref_tensor, search_tensor, gt_hm_tensor, gt_q_tensor
                        del inputs, targets
            return generator()

def load_memory_safe_dataset(h5_path, batch_size=16):
    dataset_obj = MemorySafeDataset(h5_path, batch_size)
    return dataset_obj, dataset_obj.total_samples

@tf.keras.utils.register_keras_serializable(package="Custom")
class CustomLayerQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return []
        
    def get_activations_and_quantizers(self, layer):
        return []
        
    def set_quantize_weights(self, layer, quantize_weights):
        pass
        
    def set_quantize_activations(self, layer, quantize_activations):
        pass
        
    def get_output_quantizers(self, layer):
        return [
            tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
                num_bits=8, per_axis=False, symmetric=False, narrow_range=False
            )
        ]
        
    def get_config(self):
        return {}

def main():
    parser = argparse.ArgumentParser(description="Apply QAT to Tracker Ver 4 Keras model.")
    parser.add_argument(
        "--keras_in",
        required=True,
        help="Path to input pre-trained float32 Keras model (.keras)"
    )
    parser.add_argument(
        "--keras_out",
        required=True,
        help="Path to save QAT-optimized Keras model"
    )
    parser.add_argument(
        "--train_mode",
        default="teacher-student",
        choices=["teacher-student", "dataset_labels"],
        help="QAT training mode: teacher-student (distillation) or dataset_labels (direct targets)"
    )
    parser.add_argument(
        "--h5_dataset",
        required=True,
        help="Path to compiled dataset.h5 file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs to run QAT fine-tuning (default: 2)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for fine-tuning (default: 16)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for QAT training (default: 1e-5)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of samples to train on per epoch (default: all)"
    )
    
    args = parser.parse_args()
    
    # 1. Imports and environment checks
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    sys.path.append(os.path.dirname(script_dir))
    
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError:
        print("[ERROR] tensorflow_model_optimization is not installed. Please install it first.", file=sys.stderr)
        sys.exit(1)
        
    try:
        import tracker_model
    except ImportError:
        print("[ERROR] Could not import tracker_model.py.", file=sys.stderr)
        sys.exit(1)
        
    # Restore original Conv2D and DepthwiseConv2D classes if they were overridden by tracker_model
    import tensorflow.keras.layers as keras_layers
    from tensorflow_model_optimization.python.core.keras.compat import keras as tfmot_keras
    if hasattr(tracker_model, '_orig_conv2d'):
        tf.keras.layers.Conv2D = tracker_model._orig_conv2d
        keras_layers.Conv2D = tracker_model._orig_conv2d
        tfmot_keras.layers.Conv2D = tracker_model._orig_conv2d
    if hasattr(tracker_model, '_orig_dw_conv2d'):
        tf.keras.layers.DepthwiseConv2D = tracker_model._orig_dw_conv2d
        keras_layers.DepthwiseConv2D = tracker_model._orig_dw_conv2d
        tfmot_keras.layers.DepthwiseConv2D = tracker_model._orig_dw_conv2d
        
    # 2. Load the pre-trained float32 model
    print(f"[*] Loading pre-trained model from: {args.keras_in} ...")
    custom_objects = {
        "DepthwiseCorrelationFusion": tracker_model.DepthwiseCorrelationFusion,
        "DepthToSpace": tracker_model.DepthToSpace,
        "HeatmapNormalization": tracker_model.HeatmapNormalization,
        "CustomLayerQuantizeConfig": CustomLayerQuantizeConfig,
    }
    
    with tfmot.quantization.keras.quantize_scope(custom_objects):
        teacher_model = tf.keras.models.load_model(args.keras_in, compile=False, safe_mode=False, custom_objects=custom_objects)
        
        # 3. Create the QAT annotated model
        print("[*] Annotating model layers for QAT...")
        
        def annotate_layer(layer):
            if any(k in layer.name for k in ["quality", "predicted_quality"]):
                return layer
            if layer.__class__.__name__ in ("Conv2D", "DepthwiseConv2D", "Dense"):
                return tfmot.quantization.keras.quantize_annotate_layer(layer)
            elif layer.__class__.__name__ in (
                "DepthwiseCorrelationFusion",
                "DepthToSpace",
                "HeatmapNormalization",
                "UpSampling2D",
                "AveragePooling2D",
                "MaxPooling2D",
                "Concatenate",
                "Add"
            ):
                return tfmot.quantization.keras.quantize_annotate_layer(layer, CustomLayerQuantizeConfig())
            # Check if it is a Functional sub-model
            if isinstance(layer, tf.keras.Model):
                return tf.keras.models.clone_model(layer, clone_function=annotate_layer)
            return layer
            
        # Clone model and apply annotations
        qat_annotated = tf.keras.models.clone_model(teacher_model, clone_function=annotate_layer)
        
        # Copy weights from the loaded pre-trained model to the annotated structure
        print("[*] Transferring pre-trained weights to annotated model...")
        qat_annotated.set_weights(teacher_model.get_weights())
        
        # Instantiate the QAT model
        print("[*] Applying quantization wrappers...")
        qat_model = tfmot.quantization.keras.quantize_apply(qat_annotated)
    
    # 4. Load dataset
    print(f"[*] Loading calibration dataset from: {args.h5_dataset} ...")
    dataset_loader, num_samples = load_memory_safe_dataset(args.h5_dataset, args.batch_size)
    
    # Instantiate the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, jit_compile=False)
    loss_fn_heatmap = tf.keras.losses.MeanSquaredError()
    loss_fn_quality = tf.keras.losses.BinaryCrossentropy()
    
    @tf.function
    def train_step_distill(ref, search):
        # Teacher outputs (frozen float32 predictions)
        # Call model directly to ensure Keras registers moving average updates
        t_heatmap_val, t_quality_val = teacher_model([ref, search], training=False)
        t_heatmap = tf.stop_gradient(t_heatmap_val)
        t_quality = tf.stop_gradient(t_quality_val)
        
        with tf.GradientTape() as tape:
            # Student outputs (QAT predictions)
            s_heatmap, s_quality = qat_model([ref, search], training=True)
            
            # Loss calculations (distillation)
            loss_hm = loss_fn_heatmap(t_heatmap, s_heatmap)
            loss_q = loss_fn_quality(t_quality, s_quality)
            total_loss = loss_hm + 0.5 * loss_q
            
        trainable_vars = [v for v in qat_model.trainable_variables if not any(k in v.name for k in ["quality", "predicted_quality"])]
        grads = tape.gradient(total_loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        return total_loss, loss_hm, loss_q

    @tf.function
    def train_step_gt(ref, search, gt_hm, gt_q):
        with tf.GradientTape() as tape:
            # Student outputs (QAT predictions)
            s_heatmap, s_quality = qat_model([ref, search], training=True)
            
            # Loss calculations (direct labels)
            loss_hm = loss_fn_heatmap(gt_hm, s_heatmap)
            loss_q = loss_fn_quality(gt_q, s_quality)
            total_loss = loss_hm + 0.5 * loss_q
            
        trainable_vars = [v for v in qat_model.trainable_variables if not any(k in v.name for k in ["quality", "predicted_quality"])]
        grads = tape.gradient(total_loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        return total_loss, loss_hm, loss_q

    # 6. Fine-tuning Loop
    print(f"[*] Starting QAT fine-tuning (Mode: {args.train_mode}, Epochs: {args.epochs})...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        total_loss_accum = 0.0
        steps = 0
        
        # Recreate the dataset generator at the start of each epoch to prevent exhaustion
        epoch_dataset = dataset_loader.get_generator()
        
        steps_per_epoch = num_samples // args.batch_size
        if args.max_samples is not None:
            steps_per_epoch = min(steps_per_epoch, (args.max_samples + args.batch_size - 1) // args.batch_size)
            steps_per_epoch = max(1, steps_per_epoch)
            
        pbar = tqdm.tqdm(epoch_dataset, total=steps_per_epoch, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            if steps >= steps_per_epoch:
                break
            inputs, targets = batch
            ref = inputs["reference_stack"]
            search = inputs["search_frame"]
            gt_hm = targets["predicted_heatmap"]
            gt_q = targets["predicted_quality"]
            
            if args.train_mode == "teacher-student":
                loss_val, l_hm, l_q = train_step_distill(ref, search)
            else:
                loss_val, l_hm, l_q = train_step_gt(ref, search, gt_hm, gt_q)
                
            loss_val_np = float(loss_val)
            l_hm_np = float(l_hm)
            l_q_np = float(l_q)
            
            total_loss_accum += loss_val_np
            steps += 1
            
            # Update progress status
            pbar.set_postfix({
                "loss": f"{loss_val_np:.6f}",
                "hm_loss": f"{l_hm_np:.6f}",
                "q_loss": f"{l_q_np:.6f}"
            })
            
            # Memory optimization: delete local batch variables and collect garbage periodically
            del inputs, targets, batch, ref, search, gt_hm, gt_q
            del loss_val, l_hm, l_q
            if steps % 100 == 0:
                import gc
                gc.collect()
            
        print(f"--> Epoch {epoch+1} Average Loss: {total_loss_accum / steps:.6f}")
        
    # 7. Save Keras model directly (TFLiteConverter handles quantization wrappers during conversion)
    out_dir = os.path.dirname(args.keras_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    qat_model.optimizer = None
    qat_model.save(args.keras_out)
    print(f"[SUCCESS] QAT-optimized model saved to: {args.keras_out}")

if __name__ == "__main__":
    main()
