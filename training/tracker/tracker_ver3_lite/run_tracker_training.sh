# dice_bce focal

python3 tracker_model.py train \
    --dataset_dir dataset_generation_project/video_dataset/ \
    --num_of_epochs 10 \
    --lr 1e-5 \
    --loss mse \
    --eval_pkl_num 20 \
    --init_keras_file outputs/tracker.keras \
    --output outputs/tracker.keras \
    --best_train_loss_output outputs/tracker_best_train_loss.keras \
    --log_file train.log
