#!/bin/bash
# run_tracker_training.sh
# dice_bce focal focal_dice mse centernet_focal centernet_dice

mkdir -p outputs

# Note: Batch size is determined by how the dataset was batched during the 
# create_batched_dataset.py step. (We recommend batch_size of 4 inside 
# the docker container to avoid GPU OOM issues).
# --dataset_dir dataset_generator/dataset \

python3 tracker_model.py train \
    --dataset_dir dataset_generator/dataset \
    --num_of_epochs 10 \
    --lr 1e-3 \
    --loss focal_dice \
    --eval_pkl_num 1 \
    --output outputs/tracker.keras \
    --init_keras_file outputs/tracker.keras \
    --best_train_loss_output outputs/tracker_best_train_loss.keras \
    --log_file outputs/train.log
