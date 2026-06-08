#!/bin/bash
# run_tracker_training.sh for TargetTrackerVerPixel (Google Siamese)

# =====================================================================
# Available Loss Configurations
# =====================================================================
# Heatmap Loss Options (--loss_heatmap):
#   - adaptive_wing : State-of-the-art Adaptive Wing Loss (default)
#   - dbsz_relu     : Continuous ReLU balanced MSE Loss
#   - dbsz_soft     : Soft Polynomial continuous balanced loss
#   - dbsz_hard     : Hard threshold zone masked balance loss
#   - centernet     : CenterNet penalty-reduced focal loss
#   - mse           : Standard Mean Squared Error
#
# Quality Loss Options (--loss_quality):
#   - bce           : Binary Crossentropy (default)
#   - mse           : Mean Squared Error
#   - huber         : Huber Loss
#
# Training Mode Options (--train_mode):
#   - joint         : Train both heatmap and quality branches jointly (default)
#   - heatmap_only  : [STAGE 1] Freeze quality branch layers
#   - quality_only  : [STAGE 2] Freeze backbone and heatmap decoder layers
# =====================================================================

mkdir -p outputs

# --dataset_dir: Directory containing compiled batches (dataset_generator/dataset)
# --loss_heatmap adaptive_wing: Uses the adaptive wing loss for exact heatmap regression
python3 tracker_model.py train \
    --dataset_dir dataset_generator/dataset \
    --num_of_epochs 500 \
    --lr 1e-3 \
    --loss_heatmap adaptive_wing \
    --loss_quality bce \
    --train_mode joint \
    --c_bg 6.0 \
    --eval_pkl_num 20 \
    --output outputs/tracker_pxl.keras \
    --best_train_loss_output outputs/tracker_pxl_best_train.keras \
    --log_file outputs/train.log \
    --train_mode heatmap_only
