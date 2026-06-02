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
#   - centernet     : CenterNet Penalty-reduced Focal Loss (suppresses background penalty)
#   - centernet_dice: Combination of CenterNet Focal Loss and Soft Dice Loss
#   - adaptive_wing : State-of-the-art Adaptive Wing Loss for exact heatmap regression
#
#   - dbsz_hard     : Dynamic Balanced Semantic Zone Loss (Hard Threshold masking)
#                     - Uses tf.cast thresholds (Peak: >=0.5 L1, Background: <=0.01 L2)
#                     - Completely neglects the transition zone (weight = 0.0)
#                     - Perfectly balances peak and background 50-50
#
#   - dbsz_soft     : Dynamic Balanced Semantic Zone Loss (Soft Polynomial weighting)
#                     - Uses power scaling: w_high = target^4, w_low = (1-target)^16
#                     - Transition zone naturally decays to near-zero smoothly
#                     - Perfectly balanced, continuous, and highly recommended!
#
#   - dbsz_relu     : Dynamic Balanced Semantic Zone Loss (Continuous ReLU balanced MSE)
#                     - Defines active zones via Relu(target - 0.5) and Relu(0.2 - target)
#                     - Dynamically scales by balance factors k1 and k
#                     - Calculates continuous MSE loss at the end
#
# Quality Loss Options (--loss_quality):
#   - bce           : Binary Crossentropy (standard classification/regression entropy)
#   - mse           : Mean Squared Error (penalizes large outlier score discrepancies)
#   - huber         : Huber Loss (smooth L1, robust to outlier quality ratings)
#   - logcosh       : Logarithm of hyperbolic cosine (behaves like L2 near 0, L1 far away)
#
# Training Mode Options (--train_mode):
#   - heatmap_only  : [STAGE 1] Freeze quality branch layers. Globs ONLY batch_pos_*.pkl 
#                     to train spatial features on positive-only data without negative/jitter pollution.
#   - quality_only  : [STAGE 2] Freeze shared encoders/decoder layers. Globs ONLY 
#                     batch_with_negative_*.pkl (50-50 positive/negative balanced) to train 
#                     quality classification boundaries based on frozen static features.
#   - joint         : Train both branches jointly (default).
# =====================================================================

mkdir -p outputs

# Note: Batch size is determined by how the dataset was batched during the 
# create_batched_dataset.py step.
python3 tracker_model.py train \
    --dataset_dir dataset_generator/dataset \
    --num_of_epochs 10 \
    --lr 1e-3 \
    --loss_heatmap dbsz_hard \
    --loss_quality bce \
    --train_mode joint \
    --eval_pkl_num 1 \
    --output outputs/tracker.keras \
    --init_keras_file outputs/tracker.keras \
    --best_train_loss_output outputs/tracker_best_train_loss.keras \
    --log_file outputs/train.log \
    --train_mode heatmap_only
