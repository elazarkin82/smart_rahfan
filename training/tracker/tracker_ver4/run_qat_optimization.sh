#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -e

# Change directory to script folder location for robust execution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# CONFIGURATION VARIABLES (Edit these to customize QAT & TFLite conversion)
# ==============================================================================

# --- QAT (Quantization-Aware Training) Config ---

# Path to input pre-trained float32 Keras model
KERAS_IN="outputs/tracker_coords_2_fbn.keras"

# Path to save QAT-optimized Keras model containing fake quantization parameters
# (This model is stripped of tfmot training wrappers and is ready for conversion)
KERAS_QAT_OUT="outputs/tracker_coords_2_qat.keras"

# QAT Training Mode:
#   - "teacher-student" : Knowledge Distillation (student learns to match the teacher's float32 outputs)
#   - "dataset_labels"  : Direct supervised training against HDF5 dataset targets
QAT_TRAIN_MODE="teacher-student"

# Path to the compiled HDF5 dataset (.h5) used for distillation / training inputs
H5_DATASET="dataset_generator/compiled/dataset.h5"

# Number of epochs to run QAT fine-tuning (usually 1-3 is enough to adapt weights to quantization noise)
QAT_EPOCHS=5

# Batch size for QAT training
QAT_BATCH_SIZE=4

# Learning rate for QAT (should be very small, e.g. 1e-5 or 1e-6, to avoid damaging pre-trained weights)
QAT_LR=1e-5

# Output ops to optimize during QAT.
# Only outputs listed here will contribute to the loss; variables of other output branches are frozen.
# Names are validated against the model's actual output names at runtime — a typo causes an error.
#
# Loss function used per output (auto-selected by output tensor shape):
#   - Spatial heatmap outputs (H×W > 4) → SoftArgmaxCoordLoss
#       Differentiable soft-argmax extracts [y, x] coordinates from the predicted heatmap,
#       then a Huber loss is applied against the GT peak coords. Positive-sample masking
#       based on whether the teacher heatmap contains a real peak (peak > 0.1).
#   - Scalar / small outputs               → BinaryCrossentropy (default fallback: MSE)
#
# Set to empty string to optimize ALL outputs (e.g. for final production QAT runs):
#   OUTPUT_OPS=""
# Optimize only the heatmap output (quality branch may degrade — acceptable for intermediate runs):
OUTPUT_OPS="predicted_heatmap"


# --- TFLite Converter Config ---

# Path to save the compiled static TFLite model (.tflite)
TFLITE_OUT="outputs/tracker_coord_2_qat.tflite"

# Quantization type for TFLite conversion:
#   - "none"    : Standard FP32 float model (no quantization)
#   - "fp16"    : Float16 quantization (best for GPU delegate acceleration)
#   - "int8"    : Full INT8 integer model (float32 inputs/outputs, simplifies integration)
#   - "int8_io" : Pure INT8 integer model (int8 inputs/outputs, required by Edge TPU / Coral)
TFLITE_QUANT="int8"

# Optional: Add "--copy_to_android" to copy the output TFLite model directly to the Android assets directory
COPY_TO_ANDROID="" # set to "--copy_to_android" to enable

# ==============================================================================

echo "[*] Running Quantization-Aware Training (QAT)..."
python3 utils/quantization_optimization.py \
  --keras_in "$KERAS_IN" \
  --keras_out "$KERAS_QAT_OUT" \
  --train_mode "$QAT_TRAIN_MODE" \
  --h5_dataset "$H5_DATASET" \
  --epochs "$QAT_EPOCHS" \
  --batch_size "$QAT_BATCH_SIZE" \
  --lr "$QAT_LR" \
  --max_samples 4000 \
  ${OUTPUT_OPS:+--output_ops "$OUTPUT_OPS"}

echo "[*] Converting QAT-optimized model to static TFLite..."
python3 utils/convert_to_tflite_static.py \
  --keras_in "$KERAS_QAT_OUT" \
  --tflite_out "$TFLITE_OUT" \
  --quant "$TFLITE_QUANT" \
  --h5_dataset "$H5_DATASET" \
  $COPY_TO_ANDROID

echo "[SUCCESS] QAT optimization and TFLite conversion finished successfully!"
