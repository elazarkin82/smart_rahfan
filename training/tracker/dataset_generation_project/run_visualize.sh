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
FEATURE_TYPE="asift"

# Automatically adapt threshold ratio based on algorithm selection to optimize detections
if [ "$FEATURE_TYPE" = "surf" ] || [ "$FEATURE_TYPE" = "sift" ]; then
    # Standard feature types can use a slightly looser ratio test to maximize matches
    RATIO=0.90
    MIN_INLIERS=6
else
    # ASIFT simulates multiple viewpoints and produces many matches, so 0.85 is ideal
    RATIO=0.85
    MIN_INLIERS=6
fi

echo "=========================================================="
echo "Starting Dataset Generator in Visualization Mode"
echo "Videos Source Directory: ${VIDEOS_DIR}"
echo "Feature Extraction: ${FEATURE_TYPE} (ACTIVE)"
echo "----------------------------------------------------------"
echo "Applied Permissive Thresholds:"
echo " - Lowe's Ratio (--ratio): ${RATIO}"
echo " - Min RANSAC Inliers (--min_inliers): ${MIN_INLIERS}"
echo " - RANSAC Error Thresh (--ransac_thresh): 5.0"
echo " - Min Motion Prev->Curr (--min_motion_pc): 1.0 px"
echo " - Min Motion Hist->Prev (--min_motion_hp): 3.0 px"
echo " - Min Contrast/Texture (--min_texture_std): 3.0"
echo "=========================================================="

# Unified Python Execution Call (Widescreen 3x2 HUD mode active)
python3 "${SCRIPT_DIR}/dataset_generator_from_video.py" "${VIDEOS_DIR}" \
    --output_dir "${SCRIPT_DIR}/video_dataset" \
    --feature_type "${FEATURE_TYPE}" \
    --ratio "${RATIO}" \
    --min_inliers "${MIN_INLIERS}" \
    --ransac_thresh 5.0 \
    --min_motion_pc 1.0 \
    --min_motion_hp 3.0 \
    --min_texture_std 3.0 \
    --mask_type gaussian \
    --mask_sigma 15.0 \
    --hist_radius 128 \
    --prev_radius 50 \
    --hover_prob 0.05 \
    --num_of_samples 16384 \
    --batch_size 256 \
    --visualize
