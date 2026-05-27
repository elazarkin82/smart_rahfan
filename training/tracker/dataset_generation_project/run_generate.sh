#!/bin/bash
# Exit on error
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
    FEATURE_TYPE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['feature_type'])")
    PROC_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['proc_size'])")
    RATIO=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['ratio'])")
    MIN_INLIERS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['min_inliers'])")
    RANSAC_THRESH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['ransac_thresh'])")
    MIN_MOTION_PC=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['min_motion_pc'])")
    MIN_MOTION_HP=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['min_motion_hp'])")
    MIN_TEXTURE_STD=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['min_texture_std'])")
    MIN_NCC=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['min_ncc'])")
    TARGET_MIN_MOTION_PC=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['target_min_motion_pc'])")
    TARGET_MIN_MOTION_HP=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['target_min_motion_hp'])")
    HIST_RADIUS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['hist_radius'])")
    PREV_RADIUS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['prev_radius'])")
    LABEL_RADIUS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['label_radius'])")
else
    echo "[WARNING] pipeline_config.json not found! Falling back to standard generator defaults."
    FEATURE_TYPE="sift"
    PROC_SIZE=1000
    RATIO=0.75
    MIN_INLIERS=8
    RANSAC_THRESH=1.0
    MIN_MOTION_PC=0.0
    MIN_MOTION_HP=1.0
    MIN_TEXTURE_STD=0.0
    MIN_NCC=0.9
    TARGET_MIN_MOTION_PC=0.0
    TARGET_MIN_MOTION_HP=10.0
    HIST_RADIUS=32
    PREV_RADIUS=32
    LABEL_RADIUS=32
fi

echo "=========================================================="
echo "Starting Dataset Generator in Two-Stage Staging Mode"
echo "Videos Source Directory: ${VIDEOS_DIR}"
echo "Feature Extraction: ${FEATURE_TYPE}"
echo "----------------------------------------------------------"
echo "Loaded Central configurations from pipeline_config.json:"
echo " - Processing Resolution (--proc_size): ${PROC_SIZE} px"
echo " - Lowe's Ratio (--ratio): ${RATIO}"
echo " - Min RANSAC Inliers (--min_inliers): ${MIN_INLIERS}"
echo " - RANSAC Error Thresh (--ransac_thresh): ${RANSAC_THRESH}"
echo " - Min Motion (Keypoints) Prev->Curr (--min_motion_pc): ${MIN_MOTION_PC} px"
echo " - Min Motion (Keypoints) Hist->Prev (--min_motion_hp): ${MIN_MOTION_HP} px"
echo " - Min Contrast/Texture (--min_texture_std): ${MIN_TEXTURE_STD}"
echo " - Min NCC Template Match (--min_ncc): ${MIN_NCC}"
echo " - Required Target Motion Prev->Curr (--target_min_motion_pc): ${TARGET_MIN_MOTION_PC} px"
echo " - Required Target Motion Hist->Prev (--target_min_motion_hp): ${TARGET_MIN_MOTION_HP} px"
echo "=========================================================="

# Run python generator using the centrally loaded configurations (keeps original CLI API completely active)
python3 "${SCRIPT_DIR}/dataset_generator_from_video.py" "${VIDEOS_DIR}" \
    --output_dir "${SCRIPT_DIR}/video_dataset" \
    --feature_type "${FEATURE_TYPE}" \
    --proc_size "${PROC_SIZE}" \
    --ratio "${RATIO}" \
    --min_inliers "${MIN_INLIERS}" \
    --ransac_thresh "${RANSAC_THRESH}" \
    --min_motion_pc "${MIN_MOTION_PC}" \
    --min_motion_hp "${MIN_MOTION_HP}" \
    --target_min_motion_pc "${TARGET_MIN_MOTION_PC}" \
    --target_min_motion_hp "${TARGET_MIN_MOTION_HP}" \
    --min_texture_std "${MIN_TEXTURE_STD}" \
    --min_ncc "${MIN_NCC}" \
    --mask_type gaussian \
    --mask_sigma 15.0 \
    --hist_radius "${HIST_RADIUS}" \
    --prev_radius "${PREV_RADIUS}" \
    --label_radius "${LABEL_RADIUS}" \
    --hover_prob 0.05 \
    --num_of_samples 16384 \
    --batch_size 256
