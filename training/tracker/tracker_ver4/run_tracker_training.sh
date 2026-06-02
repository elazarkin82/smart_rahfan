#!/bin/bash
# run_tracker_training.sh

# =====================================================================
# Available Loss Configurations
# =====================================================================
# Heatmap Loss Options (--loss_heatmap):
#   - mse           : Standard Mean Squared Error (good for smooth, large blobs)
#   - dice_bce      : Combination of Soft Dice Coefficient and Binary Cross Entropy
#   - focal         : Binary Focal Loss (focuses on hard/misclassified pixels)
#   - focal_dice    : Combination of Focal Loss and Soft Dice Loss
#   - centernet     : Penalty-reduced Focal Loss from CenterNet (highly robust to imbalance)
#   - centernet_dice: Combination of CenterNet Focal Loss and Soft Dice Loss
#   - adaptive_wing : State-of-the-art Adaptive Wing Loss for exact heatmap regression
#                     (highly recommended for extremely sharp and stable target locks!)
#
# Quality Loss Options (--loss_quality):
#   - bce           : Binary Crossentropy (standard classification/regression entropy)
#   - mse           : Mean Squared Error (penalizes large outlier score discrepancies)
#   - huber         : Huber Loss (smooth L1, robust to outlier quality ratings)
#   - logcosh       : Logarithm of hyperbolic cosine (behaves like L2 near 0, L1 far away)
# =====================================================================

mkdir -p outputs

# Note: Batch size is determined by how the dataset was batched during the 
# create_batched_dataset.py step.
python3 tracker_model.py train \
    --dataset_dir dataset_generator/dataset \
    --num_of_epochs 2 \
    --lr 1e-3 \
    --loss_heatmap adaptive_wing \
    --loss_quality bce \
    --eval_pkl_num 30 \
    --output outputs/tracker.keras \
    --init_keras_file outputs/tracker.keras \
    --best_train_loss_output outputs/tracker_best_train_loss.keras \
    --log_file outputs/train.log
