#!/usr/bin/env python3
import os
# Force pure-Python protobuf implementation to resolve descriptor version conflicts on host systems
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser(description="RKNN Model Converter for Radxa Zero 3 (RK3566)")
    parser.add_argument("--config", type=str, default="configs/tracker_config.yaml", help="Path to config YAML file")
    parser.add_argument("--convert-to-onnx", action="store_true", help="Convert Keras model to ONNX format before compiling to RKNN")
    args = parser.parse_args()

    # 1. Load Configuration
    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found at: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    output_path = config["model"]["output_rknn_path"]

    # Determine input model paths from config
    onnx_path = config["model"].get("onnx_model_path")
    tflite_path = config["model"].get("tflite_model_path")
    keras_path = config["model"].get("keras_model_path")

    # 1a. Validate pre-conversion flag and paths
    if args.convert_to_onnx:
        if not keras_path:
            print("[ERROR] --convert-to-onnx flag was provided, but the configuration does not specify a keras_model_path.")
            print("Currently, only Keras models (.keras) can be converted to ONNX by this script.")
            sys.exit(1)
        
        # We will perform the conversion to a temporary ONNX file
        temp_onnx_path = "/tmp/temp_keras_model.onnx"
        print(f"--> [Pre-process] Converting Keras model '{keras_path}' to temporary ONNX model: {temp_onnx_path}...")
        try:
            import tensorflow as tf
            import tf2onnx
            
            # Load custom objects dynamically if specified in the configuration
            custom_objs = {}
            if "custom_objects" in config:
                import importlib
                for obj in config["custom_objects"]:
                    name = obj["name"]
                    module_name = obj["module"]
                    path = obj.get("path")
                    if path:
                        abs_path = os.path.abspath(path)
                        if not os.path.exists(abs_path):
                            config_dir = os.path.dirname(os.path.abspath(args.config))
                            abs_path = os.path.abspath(os.path.join(config_dir, path))
                        if abs_path not in sys.path:
                            sys.path.append(abs_path)
                    try:
                        print(f"--> [Pre-process] Dynamically importing {name} from {module_name} (path: {path})...")
                        mod = importlib.import_module(module_name)
                        cls = getattr(mod, name)
                        custom_objs[name] = cls
                    except Exception as imp_err:
                        print(f"[WARNING] Could not dynamically import {name} from {module_name}: {imp_err}")

            # Load Keras model without compilation
            keras_model = tf.keras.models.load_model(keras_path, compile=False, custom_objects=custom_objs)
            
            # Build TensorSpecs for conversion (prepending batch dimension None)
            input_specs = []
            for inp in config["inputs"]:
                # Convert shape to tuple with None batch dimension
                spec_shape = [None] + list(inp["shape"])
                input_specs.append(tf.TensorSpec(spec_shape, tf.float32, name=inp["name"]))
                
            tf2onnx.convert.from_keras(
                keras_model,
                input_signature=tuple(input_specs),
                opset=13,
                output_path=temp_onnx_path
            )
            print("[SUCCESS] Keras model successfully converted to ONNX.")
            
            # Set the ONNX path to the generated file for the RKNN loader
            onnx_path = temp_onnx_path
        except Exception as e:
            print(f"[ERROR] Failed to convert Keras model to ONNX: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # If user has a Keras model but did not specify the conversion flag, throw a helpful error
        if keras_path and not onnx_path and not tflite_path:
            print("[ERROR] Keras models are not natively supported by RKNN-Toolkit2.")
            print("Please run this script with the --convert-to-onnx flag to automatically convert it to ONNX first:")
            print(f"  python3 convert.py --config {args.config} --convert-to-onnx")
            sys.exit(1)

    input_source = onnx_path or tflite_path

    print(f"=== Starting RKNN Conversion for model: {model_name} ===")
    print(f"  Source path: {input_source}")
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

    # 3. Inspect ONNX model input shapes to get expected ranks and channel sizes
    onnx_input_ranks = {}
    onnx_input_channels = {}
    if onnx_path and os.path.exists(onnx_path):
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            for onnx_inp in onnx_model.graph.input:
                dim_len = len(onnx_inp.type.tensor_type.shape.dim)
                onnx_input_ranks[onnx_inp.name] = dim_len
                if dim_len > 1:
                    # RKNN treats the second dimension (index 1) of NCHW as the channel size
                    chan_size = onnx_inp.type.tensor_type.shape.dim[1].dim_value
                    if not isinstance(chan_size, int) or chan_size <= 0:
                        chan_size = 1
                    onnx_input_channels[onnx_inp.name] = chan_size
                else:
                    onnx_input_channels[onnx_inp.name] = 1
        except Exception as onnx_err:
            print(f"[WARNING] Could not parse ONNX model inputs for dynamic shape/channel checking: {onnx_err}")

    # 4. Configure RKNN SDK
    # Map and dynamically expand mean and std values to match expected NPU channel dimensions
    mean_vals = config["rknn_config"]["mean_values"]
    std_vals = config["rknn_config"]["std_values"]
    target_platform = config["rknn_config"]["target_platform"]
    optimization_level = config["rknn_config"]["optimization_level"]
    quant_algo = config["quantization"]["quantized_algorithm"]
    quant_type = config["quantization"]["quantized_dtype"]

    expanded_mean_values = []
    expanded_std_values = []
    for i, inp in enumerate(config["inputs"]):
        name = inp["name"]
        chan_size = onnx_input_channels.get(name, 1)
        m_list = mean_vals[i] if i < len(mean_vals) else mean_vals[0]
        s_list = std_vals[i] if i < len(std_vals) else std_vals[0]
        
        # Tile/repeat the list if only one value was provided but model expects multiple channels
        if len(m_list) == 1 and chan_size > 1:
            m_list = m_list * chan_size
        if len(s_list) == 1 and chan_size > 1:
            s_list = s_list * chan_size
            
        expanded_mean_values.append(m_list)
        expanded_std_values.append(s_list)

    print("--> Configuring RKNN settings...")
    rknn.config(
        mean_values=expanded_mean_values,
        std_values=expanded_std_values,
        target_platform=target_platform,
        optimization_level=optimization_level,
        quantized_algorithm=quant_algo,
        quantized_dtype=quant_type
    )

    # 5. Load Model depending on format
    input_names = [inp["name"] for inp in config["inputs"]]
    input_shapes = []

    for inp in config["inputs"]:
        shape = list(inp["shape"])
        name = inp["name"]
        expected_rank = onnx_input_ranks.get(name)
        if expected_rank and len(shape) < expected_rank:
            print(f"--> [Shape Adjust] Prepending batch dimension 1 to input '{name}' (shape: {shape} -> {[1] + shape}) to match ONNX rank {expected_rank}")
            shape = [1] + shape
        input_shapes.append(shape)

    if onnx_path:
        print(f"--> Loading ONNX model from: {onnx_path}...")
        ret = rknn.load_onnx(
            model=onnx_path,
            inputs=input_names,
            input_size_list=input_shapes
        )
    elif tflite_path:
        print(f"--> Loading TFLite model from: {tflite_path}...")
        ret = rknn.load_tflite(
            model=tflite_path
        )
    else:
        print("[ERROR] No valid source model path (onnx_model_path or tflite_model_path) provided in config.")
        sys.exit(1)

    if ret != 0:
        print(f"[ERROR] Failed to load model. Error code: {ret}")
        sys.exit(1)

    # 5. Build RKNN Model (with or without quantization)
    do_quant = config["quantization"]["do_quantization"]
    dataset = config["quantization"]["dataset_path"]

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
        dataset=dataset
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
