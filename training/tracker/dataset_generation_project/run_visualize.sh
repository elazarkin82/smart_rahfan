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
# FEATURE MATCHING CONFIGURATION & DETECTION QUALITY TUNING
# =========================================================================

# 1. FEATURE_TYPE: Keypoint detector and descriptor extractor algorithm.
#    - "surf": Extremely fast, excellent for sharp textures (corners and edges).
#    - "sift": Highly precise, robust to illumination and scale changes.
#    - "asift": Affine SIFT - Fully simulates camera viewpoints (tilt/rotation).
#               Most robust to out-of-plane perspective changes, but slower.
FEATURE_TYPE="asift"

# 2. PROC_SIZE: Image resolution for keypoint extraction and matching (in pixels).
#    - INCREASING (e.g. 1000): Greatly improves keypoint density and precision
#      for tiny/distant features, but increases computation time.
#    - DECREASING (e.g. 512): Speeds up processing but blurs fine textures,
#      leading to fewer matches and higher failure rates.
PROC_SIZE=1000

# 3. RATIO: Lowe's ratio test threshold for descriptor matching (Range: 0.0 to 1.0).
#    - INCREASING (e.g. 0.95): Eases the constraint. Allows less distinct features
#      to match, maximizing raw keypoint count but introducing more noise (false matches).
#    - DECREASING (e.g. 0.75): Tightens the constraint. Ensures only highly distinct,
#      unambiguous matches are accepted, preventing errors but discarding valid points
#      in low-texture regions.
RATIO=0.75

# 4. MIN_INLIERS: Minimum number of RANSAC inliers required to accept the triplet.
#    - INCREASING (e.g. 10): Guarantees high mathematical reliability of the fitted
#      epipolar geometry (prevents degenerate/collinear fits), but will fail more frames.
#    - DECREASING (e.g. 5): Accepts difficult sequences with few trackable landmarks,
#      but increases the risk of accepting a mathematically incorrect Fundamental matrix.
MIN_INLIERS=8

# 5. RANSAC_THRESH: Geometric epipolar projection error tolerance (pixels in PROC_SIZE space).
#    - INCREASING (e.g. 8.0): Eases the fit. Accommodates rolling shutter, lens distortion,
#      or fast camera movements, but allows looser geometric alignments.
#    - DECREASING (e.g. 2.0): Tightens the fit. Demands perfect mathematical alignment.
#      Ensures pristine epipolar precision but rejects sequences with slight warp.
RANSAC_THRESH=1.0

# 6. MIN_MOTION_PC & MIN_MOTION_HP: Keypoint-level minimum motion thresholds for RANSAC.
#    - Kept at 0.0 by default to allow all static background landmarks (buildings, road)
#      to be matched, providing a highly dense and stable geometry fit for the F-matrix.
MIN_MOTION_PC=0.0
MIN_MOTION_HP=1.0

# 7. MIN_TEXTURE_STD: Standard deviation filter for local patch texture.
#    - INCREASING (e.g. 5.0): Restricts keypoints to highly textured regions (high contrast edges),
#      avoiding flat areas.
#    - DECREASING (e.g. 0.0): Allows matching in smooth, low-contrast landscapes (fields, roads, fog).
MIN_TEXTURE_STD=0.0

# 8. MIN_NCC: Normalized Cross-Correlation patch similarity filter (Range: -1.0 to 1.0).
#    - INCREASING (e.g. 0.90): Extremely strict. Demands near-identical visual appearance.
#      Completely eliminates false descriptor matches (e.g., matching a car headlight to a buggy).
#      Excellent for high-quality datasets.
#    - DECREASING (e.g. 0.70): Tolerates slight perspective/lighting shifts between frames,
#      but increases the risk of accepting visual mismatches.
MIN_NCC=0.9

# =========================================================================
# DEDICATED TARGET TRACKING MOTION CONSTRAINTS
# Motion requirements applied strictly to the chosen training target path (256x256 space)
# =========================================================================

# 9. TARGET_MIN_MOTION_PC & TARGET_MIN_MOTION_HP: Target-level minimum motion distance.
#    - INCREASING (e.g. PC=1.0, HP=5.0): Guarantees the selected tracking target is highly
#      dynamic, showing significant motion across frames (ideal for training active tracker models).
#    - DECREASING (e.g. 0.0): Allows selecting near-stationary targets (useful for hovering drones).
TARGET_MIN_MOTION_PC=0.0
TARGET_MIN_MOTION_HP=10.0

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
echo " - Min Motion (Keypoints) Prev->Curr (--min_motion_pc): ${MIN_MOTION_PC} px"
echo " - Min Motion (Keypoints) Hist->Prev (--min_motion_hp): ${MIN_MOTION_HP} px"
echo " - Min Contrast/Texture (--min_texture_std): ${MIN_TEXTURE_STD}"
echo " - Min NCC Template Match (--min_ncc): ${MIN_NCC}"
echo " - Required Target Motion Prev->Curr (--target_min_motion_pc): ${TARGET_MIN_MOTION_PC} px"
echo " - Required Target Motion Hist->Prev (--target_min_motion_hp): ${TARGET_MIN_MOTION_HP} px"
echo "=========================================================="

# Unified Python Execution Call (Widescreen 3x3 HUD mode active)
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
    --hist_radius 128 \
    --prev_radius 50 \
    --hover_prob 0.05 \
    --num_of_samples 16384 \
    --batch_size 256 \
    --visualize
