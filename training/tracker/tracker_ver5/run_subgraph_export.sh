#!/bin/bash
# run_subgraph_export.sh
#
# Export static Siamese Tracker subgraphs and full models running on NPU/mobile devices.
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
echo "Exporting Static Tracker Models (Quantization Mode: $QUANT)"
echo "============================================================"

# Create outputs directory
mkdir -p outputs

#KERAS_IN=outputs/tracker_coords_qat.keras
KERAS_IN=outputs/tracker_coords_fbn.keras
#KERAS_IN=outputs/tracker_coords_best_train_loss_fbn.keras

# 1. Export Part 1: Template Stack Subgraph
echo "[*] Exporting Part 1: Template Stack Subgraph..."
python3 utils/convert_to_tflite_static.py \
    --keras_in ${KERAS_IN} \
    --tflite_out outputs/tracker_template.tflite \
    --input_tensors reference_stack \
    --output_tensors reference_target_encoder \
    --quant "$QUANT" \
    --qat

# 2. Export Part 2: Frame NPU Graph
echo "[*] Exporting Part 2: Frame NPU Tracking Subgraph..."
python3 utils/convert_to_tflite_static.py \
    --keras_in ${KERAS_IN} \
    --tflite_out outputs/tracker_frame.tflite \
    --input_tensors search_frame,reference_target_encoder \
    --output_tensors predicted_heatmap_norm,predicted_quality \
    --quant "$QUANT" \
    --qat

# 2a. Export Part 2a: Frame NPU Graph (No Quality)
echo "[*] Exporting Part 2a: Frame NPU Tracking Subgraph (No Quality)..."
python3 utils/convert_to_tflite_static.py \
    --keras_in ${KERAS_IN} \
    --tflite_out outputs/tracker_frame_no_quality.tflite \
    --input_tensors search_frame,reference_target_encoder \
    --output_tensors predicted_heatmap_norm \
    --quant "$QUANT" \
    --qat

# 3. Export Full Model Graph QAT
echo "[*] Exporting Full Model Graph QAT..."
python3 utils/convert_to_tflite_static.py \
    --keras_in ${KERAS_IN} \
    --tflite_out outputs/tracker_full_qat.tflite \
    --output_tensors predicted_heatmap_norm,predicted_quality \
    --quant "$QUANT" \
    --qat
    
# 4. Export Full Model Graph
echo "[*] Exporting Full Model Graph..."
python3 utils/convert_to_tflite_static.py \
    --keras_in outputs/tracker_model_fbn.keras \
    --tflite_out outputs/tracker_full.tflite \
    --output_tensors predicted_heatmap_norm,predicted_quality \
    --quant "$QUANT" \
    --qat

# 5. Export Split Search Backbone Subgraph (outputs skip3 and search_features)
echo "[*] Exporting Split Search Backbone Subgraph..."
python3 utils/convert_to_tflite_static.py \
    --keras_in ${KERAS_IN} \
    --tflite_out outputs/tracker_search_backbone.tflite \
    --input_tensors search_frame \
    --output_tensors "search_feature_extractor:2,search_feature_extractor:3" \
    --quant "$QUANT" \
    --qat

# 6. Export Split Decoder Subgraph (inputs corr_features and skip3)
echo "[*] Exporting Split Decoder Subgraph..."
python3 utils/convert_to_tflite_static.py \
    --keras_in ${KERAS_IN} \
    --tflite_out outputs/tracker_decoder.tflite \
    --input_tensors "depthwise_correlation_fusion,search_feature_extractor:2" \
    --output_tensors predicted_heatmap \
    --quant "$QUANT" \
    --qat

# 7. Compile each Subgraph to RKNN using converter configs for first_demo
#echo "[*] Compiling subgraphs to RKNN..."
# Move to the converter directory so that relative paths in config files resolve correctly
#cd ../../../rknn/converter
#echo "--> [1/3] Converting Template subgraph..."
#python3 convert.py --config configs/tracker_tflite_template_config.yaml
#echo "--> [2/3] Converting Search Backbone subgraph..."
#python3 convert.py --config configs/tracker_tflite_search_backbone_config.yaml
#echo "--> [3/3] Converting Heatmap Decoder subgraph..."
#python3 convert.py --config configs/tracker_tflite_decoder_config.yam
# Go back to the script directory
#cd "$SCRIPT_DIR"

echo "============================================================"
echo "[SUCCESS] All models exported and converted successfully:"
echo "  - Part 1 (Template TFLite):    outputs/tracker_template.tflite"
echo "  - Part 2 (Frame TFLite):       outputs/tracker_frame.tflite"
echo "  - Part 2a (Frame No Q TFLite):  outputs/tracker_frame_no_quality.tflite"
echo "  - Full QAT TFLite:             outputs/tracker_full_qat.tflite"
echo "  - Full Model TFLite:           outputs/tracker_full.tflite"
echo "  - Split Backbone TFLite:       outputs/tracker_search_backbone.tflite"
echo "  - Split Decoder TFLite:        outputs/tracker_decoder.tflite"
echo "============================================================"
