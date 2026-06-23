#!/bin/bash
# run_all.sh
# 
# Orchestrates:
# 1. Running model generation and RKNN conversion inside the docker container.
# 2. Compiling the matmul C++ test program on the host (aarch64).
# 3. Deploying files to the Radxa board.
# 4. Executing the validation tester on the Radxa board.
# 5. Displaying SQNR results.

set -e

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RADXA_IP="192.168.58.234"

# Dynamically resolve container name running smart-rahfan-rknn:latest
CONTAINER_NAME=$(docker ps --filter "ancestor=smart-rahfan-rknn:latest" --format "{{.Names}}" | head -n 1)
if [ -z "$CONTAINER_NAME" ]; then
    CONTAINER_NAME=$(docker ps -a --filter "ancestor=smart-rahfan-rknn:latest" --format "{{.Names}}" | head -n 1)
    if [ -z "$CONTAINER_NAME" ]; then
        echo "[ERROR] Could not find any container running or created from image smart-rahfan-rknn:latest"
        exit 1
    fi
fi
echo "[+] Detected RKNN container: $CONTAINER_NAME"

echo "=========================================================="
echo "Starting Dynamic MatMul NPU Validation Pipeline"
echo "=========================================================="

# 1. Generate model
echo "[*] Step 1: Generating Keras & ONNX model inside container ($CONTAINER_NAME)..."
docker exec -u elazarkin -w "$SCRIPT_DIR" "$CONTAINER_NAME" python3 generate_matmul_model.py

# 2. Convert to RKNN
echo "[*] Step 2: Converting ONNX model to RKNN inside container ($CONTAINER_NAME)..."
docker exec -u elazarkin -w "$SCRIPT_DIR" "$CONTAINER_NAME" python3 convert_matmul.py

# 3. Compile matmul_test.cpp on host
echo "[*] Step 3: Compiling matmul_test.cpp on host (aarch64)..."
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

g++ -O3 -std=c++11 -I. matmul_test.cpp "$LIB_RKNNRT" -o matmul_test

# 4. Deploy files to Radxa Zero 3
echo "[*] Step 4: Deploying files to Radxa Zero 3 ($RADXA_IP)..."
ssh root@$RADXA_IP "mkdir -p /tmp/matmul_dynamic_test"
scp matmul.rknn matmul_test root@$RADXA_IP:/tmp/matmul_dynamic_test/

# 5. Run on Radxa Zero 3
echo "[*] Step 5: Executing benchmark on Radxa Zero 3 NPU..."
echo "----------------------------------------------------------"
ssh root@$RADXA_IP "cd /tmp/matmul_dynamic_test && ./matmul_test matmul.rknn"
echo "----------------------------------------------------------"

echo "[+] Validation Pipeline Finished successfully!"
