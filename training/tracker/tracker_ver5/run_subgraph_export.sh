#!/bin/bash
# run_subgraph_export.sh
#
# Export static Siamese Tracker subgraphs for template stacking (Part 1)
# and per-frame tracking execution (Part 2) running on NPU/mobile devices.
#
# Usage:
#   ./run_subgraph_export.sh [--quant none|fp16|int8|int8_io]

set -e

# Change directory to the script's location to resolve relative paths correctly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default quantization mode
QUANT="none"

if [ "$1" == "--quant" ] && [ -n "$2" ]; then
    QUANT="$2"
fi

echo "============================================================"
echo "Exporting Static Tracker Subgraphs (Quantization Mode: $QUANT)"
echo "============================================================"

# Create outputs directory
mkdir -p outputs

#KERAS_IN=outputs/tracker_coords_qat.keras
KERAS_IN=outputs/tracker_coords_fbn.keras

# 1. Export Part 1: Template Encoder Graph
# Input: reference_stack (1, 64, 64, 16)
# Output: reference_target_encoder (1, 8, 8, 64)
echo "[*] Exporting Part 1: Template Stack Subgraph..."
python3 utils/convert_to_tflite_static.py \
    --keras_in ${KERAS_IN} \
    --tflite_out outputs/tracker_template.tflite \
    --input_tensors reference_stack \
    --output_tensors reference_target_encoder \
    --quant "$QUANT" \
    --qat

# 2. Export Part 2: Frame NPU Graph
# Inputs: search_frame (1, 256, 256, 1), reference_target_encoder (1, 8, 8, 64)
# Outputs: predicted_heatmap, predicted_quality
echo "[*] Exporting Part 2: Frame NPU Tracking Subgraph..."
python3 utils/convert_to_tflite_static.py \
    --keras_in ${KERAS_IN} \
    --tflite_out outputs/tracker_frame.tflite \
    --input_tensors search_frame,reference_target_encoder \
    --output_tensors predicted_heatmap,predicted_quality \
    --quant "$QUANT" \
    --qat

# 2a. Export Part 2a: Frame NPU Graph (No Quality)
# Inputs: search_frame (1, 256, 256, 1), reference_target_encoder (1, 8, 8, 64)
# Outputs: predicted_heatmap
echo "[*] Exporting Part 2a: Frame NPU Tracking Subgraph (No Quality)..."
python3 utils/convert_to_tflite_static.py \
    --keras_in ${KERAS_IN} \
    --tflite_out outputs/tracker_frame_no_quality.tflite \
    --input_tensors search_frame,reference_target_encoder \
    --output_tensors predicted_heatmap \
    --quant "$QUANT" \
    --qat

# 3. Export Full Model Graph QAT
# Inputs: reference_stack (1, 64, 64, 16), search_frame (1, 256, 256, 1)
# Outputs: predicted_heatmap, predicted_quality
echo "[*] Exporting Full Model Graph..."
python3 utils/convert_to_tflite_static.py \
    --keras_in ${KERAS_IN} \
    --tflite_out outputs/tracker_full_qat.tflite \
    --quant "$QUANT" \
    --qat
    
# 4. Export Full Model Graph
# Inputs: reference_stack (1, 64, 64, 16), search_frame (1, 256, 256, 1)
# Outputs: predicted_heatmap, predicted_quality
echo "[*] Exporting Full Model Graph..."
python3 utils/convert_to_tflite_static.py \
    --keras_in outputs/tracker_model_fbn.keras \
    --tflite_out outputs/tracker_full.tflite \
    --quant "$QUANT" \
    --qat

echo "============================================================"
echo "[SUCCESS] TFLite models exported successfully:"
echo "  - Part 1 (Template):   outputs/tracker_template.tflite"
echo "  - Part 2 (Frame NPU):   outputs/tracker_frame.tflite"
echo "  - Part 2a (Frame No Q): outputs/tracker_frame_no_quality.tflite"
echo "  - Full Graph Model:     outputs/tracker_full.tflite"
echo "============================================================"
