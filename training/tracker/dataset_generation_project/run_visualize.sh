#!/bin/bash
# Exit on error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Target videos directory name
VIDEOS_DIR_NAME="videos4dataset"

# Intelligent search for the videos4dataset directory:
# 1. In the parent directory (training/tracker/videos4dataset)
# 2. Locally in this directory (dataset_generation_project/videos4dataset)
# 3. Fallback to mounted storage path if available
# 4. Otherwise, default to parent directory and ensure it exists
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
# FEATURE MATCHING CONFIGURATION (Change FEATURE_TYPE to "surf", "sift", or "asift")
# =========================================================================
FEATURE_TYPE="sift"
PROC_SIZE=800          # Higher resolution for feature matching (preserves details, default: 800)

# Hyper-Permissive thresholds to maximize raw keypoint detections in difficult views
RATIO=0.95
MIN_INLIERS=5
RANSAC_THRESH=8.0
MIN_MOTION_PC=3.0
MIN_MOTION_HP=3.0
MIN_TEXTURE_STD=0.0

echo "=========================================================="
echo "Starting Dataset Generator in Visualization Mode"
echo "Videos Source Directory: ${VIDEOS_DIR}"
echo "Feature Extraction: ${FEATURE_TYPE} (ACTIVE)"
echo "----------------------------------------------------------"
echo "Applied Permissive Thresholds & Parameters:"
echo " - Processing Resolution (--proc_size): ${PROC_SIZE} px"
echo " - Lowe's Ratio (--ratio): ${RATIO}"
echo " - Min RANSAC Inliers (--min_inliers): ${MIN_INLIERS}"
echo " - RANSAC Error Thresh (--ransac_thresh): ${RANSAC_THRESH}"
echo " - Min Motion Prev->Curr (--min_motion_pc): ${MIN_MOTION_PC} px"
echo " - Min Motion Hist->Prev (--min_motion_hp): ${MIN_MOTION_HP} px"
echo " - Min Contrast/Texture (--min_texture_std): ${MIN_TEXTURE_STD}"
echo "=========================================================="

# Unified Python Execution Call (Widescreen 3x2 HUD mode active)
python3 "${SCRIPT_DIR}/dataset_generator_from_video.py" "${VIDEOS_DIR}" \
    --output_dir "${SCRIPT_DIR}/video_dataset" \
    --feature_type "${FEATURE_TYPE}" \
    --proc_size "${PROC_SIZE}" \
    --ratio "${RATIO}" \
    --min_inliers "${MIN_INLIERS}" \
    --ransac_thresh "${RANSAC_THRESH}" \
    --min_motion_pc "${MIN_MOTION_PC}" \
    --min_motion_hp "${MIN_MOTION_HP}" \
    --min_texture_std "${MIN_TEXTURE_STD}" \
    --mask_type gaussian \
    --mask_sigma 15.0 \
    --hist_radius 128 \
    --prev_radius 50 \
    --hover_prob 0.05 \
    --num_of_samples 16384 \
    --batch_size 256 \
    --visualize
