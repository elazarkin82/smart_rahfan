#!/bin/bash
# run_all.sh
# 
# Orchestrates:
# 1. Generating TFLite model and flat test binaries using Keras-GPU container.
# 2. Compiling the TFLite model to RKNN (FP16) using RKNN Toolkit container.
# 3. Compiling the C++ validation tester program on host (aarch64).
# 4. Deploying RKNN model, test bins, and C++ tester to Radxa board.
# 5. Executing the validation benchmark on the board NPU.

set -e

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RADXA_IP="192.168.58.234"

# 1. Dynamically resolve Keras-GPU container name
echo "[*] Detecting running Keras-GPU container..."
KERAS_CONTAINER=$(docker ps --filter "ancestor=smart-rahfan-keras-gpu:latest" --format "{{.Names}}" | head -n 1)
if [ -z "$KERAS_CONTAINER" ]; then
    KERAS_CONTAINER=$(docker ps -a --filter "ancestor=smart-rahfan-keras-gpu:latest" --format "{{.Names}}" | head -n 1)
    if [ -z "$KERAS_CONTAINER" ]; then
        echo "[ERROR] Could not find any container running or created from image smart-rahfan-keras-gpu:latest"
        exit 1
    fi
fi
echo "[+] Detected Keras-GPU container: $KERAS_CONTAINER"

# 2. Dynamically resolve RKNN Toolkit container name
echo "[*] Detecting running RKNN Toolkit container..."
RKNN_CONTAINER=$(docker ps --filter "ancestor=smart-rahfan-rknn:latest" --format "{{.Names}}" | head -n 1)
if [ -z "$RKNN_CONTAINER" ]; then
    RKNN_CONTAINER=$(docker ps -a --filter "ancestor=smart-rahfan-rknn:latest" --format "{{.Names}}" | head -n 1)
    if [ -z "$RKNN_CONTAINER" ]; then
        echo "[ERROR] Could not find any container running or created from image smart-rahfan-rknn:latest"
        exit 1
    fi
fi
echo "[+] Detected RKNN container: $RKNN_CONTAINER"

echo "=========================================================="
echo "Starting MatMul Correlation Model NPU Validation Pipeline"
echo "=========================================================="

# 3. Generate TFLite and dump flat binary inputs/outputs
echo "[*] Step 1: Generating TFLite and binary reference assets inside Keras container ($KERAS_CONTAINER)..."
docker exec -u elazarkin -w "$SCRIPT_DIR" "$KERAS_CONTAINER" python3 prepare_test_data.py

# 4. Compile TFLite to RKNN (FP16 mode)
echo "[*] Step 2: Compiling TFLite model to RKNN inside NPU container ($RKNN_CONTAINER)..."
docker exec -u elazarkin -w "$SCRIPT_DIR" "$RKNN_CONTAINER" python3 convert_model.py

# 5. Compile model_test.cpp on host
echo "[*] Step 3: Compiling C++ validation tester on host (aarch64)..."
API_HEADER="/home/elazarkin/work/projects/smart_rahfan/rknn/docker/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h"
if [ ! -f "$API_HEADER" ]; then
    echo "[ERROR] rknn_api.h header not found at local path: $API_HEADER"
    exit 1
fi
cp "$API_HEADER" ./

LIB_RKNNRT="/home/elazarkin/work/projects/smart_rahfan/rknn/docker/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"
if [ ! -f "$LIB_RKNNRT" ]; then
    echo "[ERROR] librknnrt.so not found at local path: $LIB_RKNNRT"
    exit 1
fi

g++ -O3 -std=c++11 -I. model_test.cpp "$LIB_RKNNRT" -o model_test

# 6. Deploy files to Radxa Zero 3
echo "[*] Step 4: Deploying files to Radxa Zero 3 ($RADXA_IP)..."
ssh root@$RADXA_IP "mkdir -p /tmp/matmul_corr_model_test"
scp matmul_corr.rknn ref_stack.bin search_frame.bin keras_heatmap.bin keras_quality.bin model_test root@$RADXA_IP:/tmp/matmul_corr_model_test/

# 7. Run NPU inference and SQNR analysis on Radxa Zero 3
echo "[*] Step 5: Executing NPU benchmark and SQNR analysis on Radxa Zero 3..."
echo "----------------------------------------------------------"
ssh root@$RADXA_IP "cd /tmp/matmul_corr_model_test && ./model_test matmul_corr.rknn"
echo "----------------------------------------------------------"

echo "[+] Validation Pipeline finished successfully!"
