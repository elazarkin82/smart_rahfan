import os
import sys

def main():
    print("=== Starting RKNN MatMul Model Conversion ===")
    
    onnx_path = "matmul.onnx"
    output_path = "matmul.rknn"
    
    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX model not found at: {onnx_path}")
        sys.exit(1)
        
    try:
        from rknn.api import RKNN
    except ImportError:
        print("[ERROR] rknn-toolkit2 is not installed.")
        sys.exit(1)
        
    rknn = RKNN(verbose=True)
    
    print("--> Configuring RKNN settings...")
    rknn.config(
        target_platform="rk3566",
        optimization_level=3
    )
    
    print(f"--> Loading ONNX model: {onnx_path}...")
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=["input_A", "input_B"],
        input_size_list=[[1, 128, 64], [1, 64, 128]]
    )
    if ret != 0:
        print(f"[ERROR] Failed to load model. Error code: {ret}")
        sys.exit(1)
        
    print("--> Building RKNN model (FP16 mode, no quantization)...")
    ret = rknn.build(
        do_quantization=False
    )
    if ret != 0:
        print(f"[ERROR] Failed to build model. Error code: {ret}")
        sys.exit(1)
        
    print(f"--> Exporting RKNN model to: {output_path}...")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"[ERROR] Failed to export RKNN model. Error code: {ret}")
        sys.exit(1)
        
    print("[SUCCESS] RKNN MatMul model successfully exported!")
    rknn.release()

if __name__ == "__main__":
    main()
