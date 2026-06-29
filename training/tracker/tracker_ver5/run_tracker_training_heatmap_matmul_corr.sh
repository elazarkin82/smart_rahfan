#!/bin/bash
# run_tracker_training_heatmap_matmul_corr.sh
# 
# Script to run the experimental tracker training using the matmul correlation model.

mkdir -p outputs

# TensorFlow GPU memory policy for this training process.
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export TRACKER_GPU_MEMORY_LIMIT_MB="${TRACKER_GPU_MEMORY_LIMIT_MB:-51200}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

echo "[TF ENV] TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR}"
echo "[TF ENV] TRACKER_GPU_MEMORY_LIMIT_MB=${TRACKER_GPU_MEMORY_LIMIT_MB}"

# Run training using the matmul correlation loss script
#
# --train_mode heatmap_only: Trains spatial features using coordinate distance loss on positive samples.

BATCH_SIZE=128

# --dataset_cache_mode streaming for memory error
python3 tracker_model_matmul_corr.py train \
    --dataset_dir dataset_generator/compiled \
    --num_of_epochs 30 \
    --lr 1e-3 \
    --loss_quality bce \
    --output outputs/tracker_matmul_corr.keras \
    --init_keras_file outputs/tracker_matmul_corr.keras \
    --best_train_loss_output outputs/tracker_matmul_corr_best_train_loss.keras \
    --log_file outputs/train_matmul_corr.log \
    --train_mode heatmap_only \
    --batch_size ${BATCH_SIZE} \
    --dataset_cache_mode streaming
    
python3 tracker_model_matmul_corr.py train \
    --dataset_dir dataset_generator/compiled \
    --num_of_epochs 20 \
    --lr 1e-4 \
    --loss_quality bce \
    --output outputs/tracker_matmul_corr.keras \
    --init_keras_file outputs/tracker_matmul_corr.keras \
    --best_train_loss_output outputs/tracker_matmul_corr_best_train_loss.keras \
    --log_file outputs/train_matmul_corr.log \
    --train_mode heatmap_only \
    --batch_size ${BATCH_SIZE} \
    --dataset_cache_mode streaming

python3 tracker_model_matmul_corr.py train \
    --dataset_dir dataset_generator/compiled \
    --num_of_epochs 10 \
    --lr 1e-5 \
    --loss_quality bce \
    --output outputs/tracker_matmul_corr.keras \
    --init_keras_file outputs/tracker_matmul_corr.keras \
    --best_train_loss_output outputs/tracker_matmul_corr_best_train_loss.keras \
    --log_file outputs/train_matmul_corr.log \
    --train_mode heatmap_only \
    --batch_size ${BATCH_SIZE} \
    --dataset_cache_mode streaming
