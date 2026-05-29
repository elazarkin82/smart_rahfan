python3 tracker_model.py train \
    --dataset_dir dataset_generation_project/video_dataset/ \
    --num_of_epochs 10 \
    --lr 1e-4 \
    --loss dice_bce \
    --eval_pkl_num 4 \
    --output outputs/tracker.keras \
    --best_train_loss_output outputs/tracker_best_train_loss.keras \
    --log_file train.log
