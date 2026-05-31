#!/bin/bash
# run_tracker_training.sh

mkdir -p outputs

python3 tracker_model.py train \
    --dataset_dir dataset_generator/dataset \
    --num_of_epochs 5 \
    --lr 1e-4 \
    --loss focal \
    --batch_size 16 \
    --eval_pkl_num 20 \
    --output outputs/tracker.keras \
    --init_keras_file outputs/tracker.keras \
    --best_train_loss_output outputs/tracker_best_train_loss.keras \
    --log_file outputs/train.log
