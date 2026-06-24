import os
import sys

def main():
    # Attempt importing RKNN toolkit
    try:
        from rknn.api import RKNN
    except ImportError:
        print("[ERROR] rknn-toolkit2 is not installed or not in python path.")
        print("Please run this script inside the rknn docker container.")
        sys.exit(1)

    # Load model.conf dynamically to get correct shapes
    import configparser
    config_path = "/home/elazarkin/work/projects/smart_rahfan/training/tracker/tracker_ver5/model.conf"
    config = configparser.ConfigParser()
    config.read(config_path)
    stack_layers = config.getint("Stack", "stack_layers", fallback=2)
    stack_target_size = config.getint("Stack", "stack_target_size", fallback=128)
    search_frame_size = config.getint("Stack", "search_frame_size", fallback=256)
    print(f"--> Loaded config from {config_path}: stack_layers={stack_layers}, stack_target_size={stack_target_size}, search_frame_size={search_frame_size}")

    print("--> Initializing RKNN tool...")
    rknn = RKNN(verbose=True)
    
    # Configure RKNN settings dynamically based on stack_layers
    print("--> Configuring RKNN settings (Target: rk3566, Optimization level: 3)...")
    rknn.config(
        target_platform='rk3566',
        optimization_level=3,
        mean_values=[[0] * stack_layers, [0]],
        std_values=[[255] * stack_layers, [255]]
    )
    
    tflite_model_path = 'matmul_corr.tflite'
    
    # TFLite loading is not supported on ARM64 by RKNN Toolkit2, so convert to ONNX first if on ARM64
    import platform
    is_arm64 = platform.machine() in ["aarch64", "arm64"]
    if is_arm64:
        print("[*] ARM64 platform detected. TFLite model loading is not natively supported by RKNN-Toolkit2 on ARM64.")
        print("--> Automatically converting TFLite model to ONNX first...")
        onnx_model_path = 'matmul_corr.onnx'
        try:
            import subprocess
            cmd = [
                sys.executable, "-m", "tf2onnx.convert",
                "--tflite", tflite_model_path,
                "--output", onnx_model_path,
                "--opset", "13",
                "--inputs-as-nchw", "serving_default_reference_stack:0,serving_default_search_frame:0"
            ]
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print("[SUCCESS] TFLite model successfully converted to ONNX.")
            
            # Load ONNX model
            print(f"--> Loading ONNX model: {onnx_model_path}")
            ret = rknn.load_onnx(
                model=onnx_model_path,
                inputs=["serving_default_reference_stack:0", "serving_default_search_frame:0"],
                input_size_list=[[1, stack_layers, stack_target_size, stack_target_size], [1, 1, search_frame_size, search_frame_size]]
            )
        except Exception as e:
            print(f"[ERROR] Failed to convert or load ONNX model: {e}")
            sys.exit(1)
    else:
        # Load the TFLite model directly
        print(f"--> Loading TFLite model: {tflite_model_path}")
        ret = rknn.load_tflite(model=tflite_model_path)
        
    if ret < 0:
        print(f"[ERROR] Failed to load model: {ret}")
        sys.exit(1)
        
    # Build RKNN model (do_quantization=False compiles in FP16 mode)
    print("--> Building RKNN model (do_quantization=False → FP16 Mode)...")
    ret = rknn.build(do_quantization=False)
    if ret < 0:
        print(f"[ERROR] Failed to build RKNN model: {ret}")
        sys.exit(1)
        
    # Export to RKNN format
    rknn_output_path = 'matmul_corr.rknn'
    print(f"--> Exporting RKNN model to: {rknn_output_path}")
    ret = rknn.export_rknn(rknn_output_path)
    if ret < 0:
        print(f"[ERROR] Failed to export RKNN model: {ret}")
        sys.exit(1)
        
    print(f"[SUCCESS] RKNN model successfully exported to {rknn_output_path}")

if __name__ == "__main__":
    main()
