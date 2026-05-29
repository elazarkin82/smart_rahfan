#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Target videos directory name
VIDEOS_DIR_NAME="videos4dataset"

# Intelligent search for the videos4dataset directory
if [ -d "${SCRIPT_DIR}/../${VIDEOS_DIR_NAME}" ]; then
    VIDEOS_DIR="${SCRIPT_DIR}/../${VIDEOS_DIR_NAME}"
elif [ -d "${SCRIPT_DIR}/${VIDEOS_DIR_NAME}" ]; then
    VIDEOS_DIR="${SCRIPT_DIR}/${VIDEOS_DIR_NAME}"
elif [ -d "/home/elazarkin/storage/video_samples" ]; then
    VIDEOS_DIR="/home/elazarkin/storage/video_samples"
else
    VIDEOS_DIR="${SCRIPT_DIR}/../${VIDEOS_DIR_NAME}"
    echo "Creating directory: ${VIDEOS_DIR}"
    mkdir -p "${VIDEOS_DIR}"
fi

# =========================================================================
# CENTRALIZED PARAMETER LOADING FROM pipeline_config.json
# =========================================================================
CONFIG_FILE="${SCRIPT_DIR}/pipeline_config.json"

if [ -f "$CONFIG_FILE" ]; then
    PROC_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['proc_size'])")
    RATIO=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['ratio'])")
    MIN_INLIERS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['min_inliers'])")
    RANSAC_THRESH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['ransac_thresh'])")
    MIN_TEXTURE_STD=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['min_texture_std'])")
    MIN_NCC=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['min_ncc'])")
    TARGET_MIN_MOTION=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['target_min_motion'])")
    MASK_SIGMA=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['mask_sigma'])")
    LABEL_RADIUS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['label_radius'])")
    BATCH_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['batch_size'])")
    TEMPORAL_STEP=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['temporal_step_seconds'])")
    MAX_FAILURES=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['max_consecutive_failures'])")
else
    echo "[WARNING] pipeline_config.json not found! Falling back to defaults."
    PROC_SIZE=1000
    RATIO=0.85
    MIN_INLIERS=6
    RANSAC_THRESH=5.0
    MIN_TEXTURE_STD=3.0
    MIN_NCC=0.75
    TARGET_MIN_MOTION=3.0
    MASK_SIGMA=30.0
    LABEL_RADIUS=32.0
    BATCH_SIZE=256
    TEMPORAL_STEP=1.0
    MAX_FAILURES=10
fi

echo "=========================================================="
echo "Starting Symmetric 2-Frame Dataset Generator"
echo "Videos Source Directory: ${VIDEOS_DIR}"
echo "----------------------------------------------------------"
echo "Loaded configurations from pipeline_config.json:"
echo " - Processing Resolution: ${PROC_SIZE} px"
echo " - Lowe's Ratio: ${RATIO}"
echo " - Min RANSAC Inliers: ${MIN_INLIERS}"
echo " - RANSAC Error Thresh: ${RANSAC_THRESH}"
echo " - Min Contrast/Texture: ${MIN_TEXTURE_STD}"
echo " - Min NCC Template Match: ${MIN_NCC}"
echo " - Required Target Motion: ${TARGET_MIN_MOTION} px"
echo " - Mask Sigma (Exponential): ${MASK_SIGMA}"
echo " - Label Radius: ${LABEL_RADIUS}"
echo " - Batch Size: ${BATCH_SIZE}"
echo " - Temporal Seek Step: ${TEMPORAL_STEP} s"
echo "=========================================================="

python3 "${SCRIPT_DIR}/dataset_generator_from_video.py" "${VIDEOS_DIR}" \
    --output_dir "${SCRIPT_DIR}/video_dataset" \
    --proc_size "${PROC_SIZE}" \
    --ratio "${RATIO}" \
    --min_inliers "${MIN_INLIERS}" \
    --ransac_thresh "${RANSAC_THRESH}" \
    --target_min_motion "${TARGET_MIN_MOTION}" \
    --min_texture_std "${MIN_TEXTURE_STD}" \
    --min_ncc "${MIN_NCC}" \
    --mask_sigma "${MASK_SIGMA}" \
    --label_radius "${LABEL_RADIUS}" \
    --batch_size "${BATCH_SIZE}" \
    --temporal_step_seconds "${TEMPORAL_STEP}" \
    --max_consecutive_failures "${MAX_FAILURES}" \
    --num_of_samples 16384
