#!/usr/bin/env python3
import os
import sys
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser(description="RKNN Model Converter for Radxa Zero 3 (RK3566)")
    parser.add_argument("--config", type=str, default="configs/tracker_config.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    # 1. Load Configuration
    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found at: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    onnx_path = config["model"]["onnx_model_path"]
    output_path = config["model"]["output_rknn_path"]

    print(f"=== Starting RKNN Conversion for model: {model_name} ===")
    print(f"  ONNX path: {onnx_path}")
    print(f"  Output path: {output_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 2. Initialize RKNN
    try:
        from rknn.api import RKNN
    except ImportError:
        print("[ERROR] rknn-toolkit2 is not installed or not in python path.")
        print("Please run this script inside the rknn docker container or install rknn-toolkit2.")
        sys.exit(1)

    rknn = RKNN(verbose=True)

    # 3. Configure RKNN SDK
    # Map configuration parameters
    mean_vals = config["rknn_config"]["mean_values"]
    std_vals = config["rknn_config"]["std_values"]
    target_platform = config["rknn_config"]["target_platform"]
    optimization_level = config["rknn_config"]["optimization_level"]

    print("--> Configuring RKNN settings...")
    rknn.config(
        mean_values=mean_vals,
        std_values=std_vals,
        target_platform=target_platform,
        optimization_level=optimization_level
    )

    # 4. Load ONNX Model
    input_names = [inp["name"] for inp in config["inputs"]]
    input_shapes = [inp["shape"] for inp in config["inputs"]]

    print("--> Loading ONNX model...")
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=input_names,
        input_size_list=input_shapes
    )
    if ret != 0:
        print(f"[ERROR] Failed to load ONNX model. Error code: {ret}")
        sys.exit(1)

    # 5. Build RKNN Model (with or without quantization)
    do_quant = config["quantization"]["do_quantization"]
    dataset = config["quantization"]["dataset_path"]
    quant_algo = config["quantization"]["quantized_algorithm"]
    quant_type = config["quantization"]["quantized_dtype"]

    print(f"--> Building RKNN model (Quantization: {do_quant})...")
    
    # Check dataset existence if quantization is requested
    if do_quant:
        if not os.path.exists(dataset):
            print(f"[WARNING] Calibration dataset file '{dataset}' not found.")
            print("Please create the calibration dataset text file containing representative inputs.")
            print("Proceeding without quantization to build in FP16 mode as fallback...")
            do_quant = False
            dataset = None

    ret = rknn.build(
        do_quantization=do_quant,
        dataset=dataset,
        quantized_algorithm=quant_algo,
        quantized_dtype=quant_type
    )
    if ret != 0:
        print(f"[ERROR] Failed to build RKNN model. Error code: {ret}")
        sys.exit(1)

    # 6. Export RKNN Model
    print("--> Exporting RKNN model...")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"[ERROR] Failed to export RKNN model. Error code: {ret}")
        sys.exit(1)

    print(f"[SUCCESS] RKNN model successfully exported to: {output_path}")
    rknn.release()

if __name__ == "__main__":
    main()
