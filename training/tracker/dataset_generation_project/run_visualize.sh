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

echo "=========================================================="
echo "Starting Dataset Generator in Visualization Mode"
echo "Videos Source Directory: ${VIDEOS_DIR}"
echo "Feature Extraction: ASIFT (View viewpoint simulations)"
echo "----------------------------------------------------------"
echo "Applied Permissive Thresholds:"
echo " - Lowe's Ratio (--ratio): 0.85"
echo " - Min RANSAC Inliers (--min_inliers): 6"
echo " - RANSAC Error Thresh (--ransac_thresh): 5.0"
echo " - Min Motion Prev->Curr (--min_motion_pc): 1.0 px"
echo " - Min Motion Hist->Prev (--min_motion_hp): 3.0 px"
echo " - Min Contrast/Texture (--min_texture_std): 3.0"
echo "=========================================================="

# Execute python script in visualization mode with permissive threshold tuning
python3 "${SCRIPT_DIR}/dataset_generator_from_video.py" "${VIDEOS_DIR}" \
    --feature_type asift \
    --ratio 0.85 \
    --min_inliers 6 \
    --ransac_thresh 5.0 \
    --min_motion_pc 1.0 \
    --min_motion_hp 3.0 \
    --min_texture_std 3.0 \
    --visualize
