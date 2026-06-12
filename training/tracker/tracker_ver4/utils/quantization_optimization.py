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
import numpy as np
import tqdm

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
        
    # 2. Load the pre-trained float32 model
    print(f"[*] Loading pre-trained model from: {args.keras_in} ...")
    custom_objects = {
        "SafeGroupNormalization": tracker_model.SafeGroupNormalization,
        "DepthwiseCorrelationFusion": tracker_model.DepthwiseCorrelationFusion,
        "DepthToSpace": tracker_model.DepthToSpace,
        "HeatmapNormalization": tracker_model.HeatmapNormalization,
    }
    teacher_model = tf.keras.models.load_model(args.keras_in, compile=False, safe_mode=False, custom_objects=custom_objects)
    
    # 3. Create the QAT annotated model
    print("[*] Annotating model layers for QAT...")
    
    def annotate_layer(layer):
        # Only annotate standard weight/trainable layers to avoid breaking custom layer graphs
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Dense)):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        # Check if it is a Functional sub-model
        if isinstance(layer, tf.keras.Model):
            return tf.keras.models.clone_model(layer, clone_function=annotate_layer)
        return layer
        
    # Clone model and apply annotations
    qat_annotated = tf.keras.models.clone_model(teacher_model, clone_function=annotate_layer)
    
    # Instantiate the QAT model
    print("[*] Applying quantization wrappers...")
    qat_model = tfmot.quantization.keras.quantize_apply(qat_annotated)
    
    # 4. Load dataset
    print(f"[*] Loading calibration dataset from: {args.h5_dataset} ...")
    dataset = tracker_model.load_hdf5_dataset(args.h5_dataset, args.batch_size, is_val=False)
    
    # 5. Define QAT optimization loss and train steps
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn_heatmap = tf.keras.losses.MeanSquaredError()
    loss_fn_quality = tf.keras.losses.BinaryCrossEntropy()
    
    @tf.function
    def train_step_distill(ref, search):
        # Teacher outputs (frozen float32 predictions)
        t_heatmap, t_quality = teacher_model([ref, search], training=False)
        
        with tf.GradientTape() as tape:
            # Student outputs (QAT predictions)
            s_heatmap, s_quality = qat_model([ref, search], training=True)
            
            # Loss calculations (distillation)
            loss_hm = loss_fn_heatmap(t_heatmap, s_heatmap)
            loss_q = loss_fn_quality(t_quality, s_quality)
            total_loss = loss_hm + 0.5 * loss_q
            
        grads = tape.gradient(total_loss, qat_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, qat_model.trainable_variables))
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
            
        grads = tape.gradient(total_loss, qat_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, qat_model.trainable_variables))
        return total_loss, loss_hm, loss_q

    # 6. Fine-tuning Loop
    print(f"[*] Starting QAT fine-tuning (Mode: {args.train_mode}, Epochs: {args.epochs})...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        total_loss_accum = 0.0
        steps = 0
        
        # Stream over training samples
        pbar = tqdm.tqdm(dataset, desc=f"Epoch {epoch+1}")
        for (ref, search), (gt_hm, gt_q) in pbar:
            if args.train_mode == "teacher-student":
                loss_val, l_hm, l_q = train_step_distill(ref, search)
            else:
                loss_val, l_hm, l_q = train_step_gt(ref, search, gt_hm, gt_q)
                
            total_loss_accum += loss_val.numpy()
            steps += 1
            
            # Update progress status
            pbar.set_postfix({
                "loss": f"{loss_val.numpy():.6f}",
                "hm_loss": f"{l_hm.numpy():.6f}",
                "q_loss": f"{l_q.numpy():.6f}"
            })
            
        print(f"--> Epoch {epoch+1} Average Loss: {total_loss_accum / steps:.6f}")
        
    # 7. Strip quantization wrappers and save Keras model
    print("[*] Stripping quantization wrappers for TFLite compliance...")
    stripped_model = tfmot.quantization.keras.quantize_strip(qat_model)
    
    # Save the resulting model
    out_dir = os.path.dirname(args.keras_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    stripped_model.save(args.keras_out)
    print(f"[SUCCESS] QAT-optimized model saved to: {args.keras_out}")

if __name__ == "__main__":
    main()
