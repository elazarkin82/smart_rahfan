#!/bin/bash
# run_tracker_training_coords.sh
# 
# Script to run the experimental tracker training using the coordinate-based loss.
# This script executes tracker_model_coords.py, which uses Soft-Argmax to extract coordinates 
# and computes the loss directly on the Euclidean distance from the true target center.

# =====================================================================
# Available Loss Configurations
# =====================================================================
# Heatmap Loss:
#   * Note: In this experimental model, normal pixel-level heatmap loss functions (like 
#     adaptive_wing or dbsz) are not used. Instead, coordinate_distance_loss (Huber Loss 
#     on the coordinates extracted via Soft-Argmax) is computed. All dbsz parameters 
#     (--loss_heatmap, --c_bg, --dbsz_border) are completely omitted.
#
# Quality Loss Options (--loss_quality):
#   - bce           : Binary Crossentropy (standard classification/regression entropy)
#   - mse           : Mean Squared Error (penalizes large outlier score discrepancies)
#   - huber         : Huber Loss (smooth L1, robust to outlier quality ratings)
#   - logcosh       : Logarithm of hyperbolic cosine (behaves like L2 near 0, L1 far away)
#
# Training Mode Options (--train_mode):
#   - heatmap_only  : [STAGE 1] Freeze quality branch layers. Trains spatial features 
#                     using only coordinate distance loss (Soft-Argmax) on positive samples.
#   - quality_only  : [STAGE 2] Freeze shared encoders/decoder layers. Trains quality 
#                     classification boundaries using dynamically generated targets 
#                     (linear decay based on peak coordinate distance).
#   - joint         : Train both branches jointly (default).
# =====================================================================

mkdir -p outputs

# Run training using the coordinate-based loss script
#
# --train_mode joint: Trains both the backbone (via Soft-Argmax coordinate loss) and the quality head.
# --loss_quality bce: Uses Binary Crossentropy for the quality branch.
# --batch_size 32: Sets batch size for training.
python3 tracker_model_coords.py train \
    --dataset_dir dataset_generator/compiled \
    --num_of_epochs 80 \
    --lr 1e-5 \
    --loss_quality bce \
    --output outputs/tracker_coords_2.keras \
    --init_keras_file outputs/tracker_coords.keras \
    --best_train_loss_output outputs/tracker_coords_2_best_train_loss.keras \
    --log_file outputs/train_coords.log \
    --train_mode heatmap_only \
    --batch_size 32
