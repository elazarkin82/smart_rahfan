#!/usr/bin/env python3
"""
TargetTrackerVer4 Static TFLite Converter
=========================================
This script reconstructs the Siamese-Attention architecture of Tracker Ver 4,
loads only the trained weights from a Keras model (.keras), and exports it
as a highly-optimized static TFLite model (batch size = 1).

This completely avoids:
  1. Keras version mismatch errors (Keras 3 vs Keras 2 'tf_keras' namespace errors).
  2. Python version marshal/bytecode compatibility errors with Lambda layers.
  3. TFLite dynamic shape and BROADCAST_TO runtime exceptions on Android.

Usage:
    python3 convert_to_tflite_static.py --keras_in outputs/tracker.keras --tflite_out outputs/tracker.tflite --copy_to_android
"""

import os
import sys
import argparse
import shutil
import numpy as np

def transpose_preserves_memory_layout(shape, perm):
    if perm is None:
        perm = list(range(len(shape)))[::-1]
    non_singleton_indices = [i for i, dim in enumerate(shape) if dim is None or dim > 1]
    permuted_non_singleton = [idx for idx in perm if idx in non_singleton_indices]
    return permuted_non_singleton == non_singleton_indices

def find_tensor_by_name(model, name):
    tensor_index = 0
    if ":" in name:
        try:
            tensor_index = int(name.split(":")[-1])
        except ValueError:
            pass
            
    base_name = name.split(":")[0]
    parts = base_name.split("/")
    
    def search_layers(m, layer_name):
        for l in m.layers:
            if l.name == layer_name:
                return l
            if hasattr(l, "layers"):
                res = search_layers(l, layer_name)
                if res is not None:
                    return res
        return None

    target_layer_name = parts[-1]
    target_layer = search_layers(model, target_layer_name)
    
    if target_layer is not None:
        try:
            if hasattr(target_layer, "_inbound_nodes") and len(target_layer._inbound_nodes) > 0:
                node_outputs = target_layer._inbound_nodes[0].output_tensors
                if isinstance(node_outputs, list):
                    if len(parts) > 1:
                        for out in node_outputs:
                            if name in out.name or out.name.endswith(name):
                                return out
                    if tensor_index < len(node_outputs):
                        return node_outputs[tensor_index]
                    return node_outputs[0]
                return node_outputs
        except Exception:
            pass
        try:
            if isinstance(target_layer.output, list):
                if len(parts) > 1:
                    for out in target_layer.output:
                        if name in out.name or out.name.endswith(name):
                            return out
                if tensor_index < len(target_layer.output):
                    return target_layer.output[tensor_index]
                return target_layer.output[0]
            return target_layer.output
        except Exception:
            pass
            
    def search_outputs(m):
        for l in m.layers:
            try:
                if isinstance(l.output, list):
                    for out in l.output:
                        if out.name == name or out.name.startswith(name) or name in out.name:
                            return out
                else:
                    if l.output.name == name or l.output.name.startswith(name) or name in l.output.name:
                        return l.output
            except Exception:
                pass
            if hasattr(l, "layers"):
                res = search_outputs(l)
                if res is not None:
                    return res
        return None
        
    res_tensor = search_outputs(model)
    if res_tensor is not None:
        return res_tensor
        
    raise ValueError(f"Could not find tensor or layer with name '{name}' in model.")

def build_subgraph(model, input_names, output_names):
    import tensorflow as tf
    from tensorflow.keras import layers
    
    def map_tensors_in_args_kwargs(args, kwargs, tensor_map):
        def map_val(val):
            if isinstance(val, list):
                return [map_val(v) for v in val]
            if isinstance(val, tuple):
                return tuple(map_val(v) for v in val)
            if isinstance(val, dict):
                return {k: map_val(v) for k, v in val.items()}
            if id(val) in tensor_map:
                return tensor_map[id(val)]
            return val

        new_args = map_val(args)
        new_kwargs = map_val(kwargs)
        return new_args, new_kwargs
        
    target_inputs = []
    for name in input_names:
        tensor = find_tensor_by_name(model, name)
        target_inputs.append(tensor)
        
    target_outputs = []
    for name in output_names:
        tensor = find_tensor_by_name(model, name)
        target_outputs.append(tensor)
        
    tensor_map = {}
    new_inputs = []
    
    for tensor in target_inputs:
        is_input = False
        try:
            if hasattr(tensor, "_keras_history"):
                layer, _, _ = tensor._keras_history
                if isinstance(layer, tf.keras.layers.InputLayer):
                    is_input = True
        except Exception:
            pass
            
        if is_input:
            new_inputs.append(tensor)
            tensor_map[id(tensor)] = tensor
        else:
            shape = tensor.shape.as_list()
            input_name = tensor.name.replace(":", "_").replace("/", "_")
            new_in = layers.Input(batch_shape=shape, dtype=tensor.dtype, name=f"input_{input_name}")
            new_inputs.append(new_in)
            tensor_map[id(tensor)] = new_in
            print(f"[Subgraph] Created new input placeholder '{new_in.name}' for tensor '{tensor.name}' with shape {shape}")

    all_layers = model.layers
    
    for layer in all_layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
            
        try:
            nodes = layer._inbound_nodes
            for node_idx, node in enumerate(nodes):
                node_inputs = node.input_tensors
                if not isinstance(node_inputs, (list, tuple)):
                    node_inputs = [node_inputs]
                    
                if all(id(inp) in tensor_map for inp in node_inputs):
                    mapped_inps = [tensor_map[id(inp)] for inp in node_inputs]
                    if len(mapped_inps) == 1:
                        mapped_inps = mapped_inps[0]
                    if hasattr(node, "call_args") and node.call_args is not None:
                        new_args, new_kwargs = map_tensors_in_args_kwargs(node.call_args, node.call_kwargs, tensor_map)
                        new_out = layer(*new_args, **new_kwargs)
                    else:
                        new_out = layer(mapped_inps)
                    node_outputs = node.output_tensors
                    
                    # Handle list vs single tensor matching safely
                    if isinstance(node_outputs, (list, tuple)):
                        if not isinstance(new_out, (list, tuple)):
                            new_out = [new_out]
                        for orig_o, new_o in zip(node_outputs, new_out):
                            tensor_map[id(orig_o)] = new_o
                    else:
                        tensor_map[id(node_outputs)] = new_out
                    break
        except Exception:
            try:
                if isinstance(layer.input, list):
                    layer_inputs = layer.input
                else:
                    layer_inputs = [layer.input]
                if all(id(inp) in tensor_map for inp in layer_inputs):
                    mapped_inps = [tensor_map[id(inp)] for inp in layer_inputs]
                    if len(mapped_inps) == 1:
                        mapped_inps = mapped_inps[0]
                    new_out = layer(mapped_inps)
                    if isinstance(layer.output, list):
                        if not isinstance(new_out, (list, tuple)):
                            new_out = [new_out]
                        for orig_o, new_o in zip(layer.output, new_out):
                            tensor_map[id(orig_o)] = new_o
                    else:
                        tensor_map[id(layer.output)] = new_out
            except Exception:
                pass
        
    missing_outputs = [out for out in target_outputs if id(out) not in tensor_map]
    if missing_outputs:
        raise ValueError(f"Could not map output tensors: {[t.name for t in missing_outputs]}")
        
    subgraph_model = tf.keras.models.Model(
        inputs=new_inputs, 
        outputs=[tensor_map[id(out)] for out in target_outputs],
        name=model.name + "_subgraph"
    )
    return subgraph_model, target_inputs

def optimize_and_trim_model(model, trim_outputs=True):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # 1. Replace AveragePooling2D (stride=1) layers with DepthwiseConv2D
    # to avoid NPU-to-CPU fallback.
    has_avg_pool = any(isinstance(l, layers.AveragePooling2D) and l.strides == (1, 1) for l in model.layers)
    if has_avg_pool:
        print("[Optimizer] Found AveragePooling2D with stride=1. Reconstructing model graph to replace with DepthwiseConv2D...")
        try:
            tensor_map = {}
            new_inputs = []
            for inp in model.inputs:
                new_inp = layers.Input(shape=inp.shape[1:], dtype=inp.dtype, name=inp.name.split(":")[0])
                new_inputs.append(new_inp)
                tensor_map[id(inp)] = new_inp
                
            for layer in model.layers:
                if isinstance(layer, layers.InputLayer):
                    continue
                
                # Retrieve inbound node connections
                inbound_nodes = layer.inbound_nodes if hasattr(layer, "inbound_nodes") else layer._inbound_nodes
                if not inbound_nodes:
                    continue
                input_tensors = inbound_nodes[0].input_tensors
                if isinstance(input_tensors, list):
                    layer_inputs = [tensor_map[id(t)] for t in input_tensors]
                else:
                    layer_inputs = tensor_map[id(input_tensors)]
                    
                if isinstance(layer, layers.AveragePooling2D) and layer.strides == (1, 1):
                    pool_size = layer.pool_size
                    padding = layer.padding
                    name = layer.name
                    
                    # Find channel size
                    in_shape = layer.input_spec.shape if (hasattr(layer, "input_spec") and layer.input_spec) else layer.input.shape
                    channels = in_shape[-1] if in_shape is not None else 1
                    
                    # Compute constant weight scale
                    kh, kw = pool_size
                    scale = 1.0 / (kh * kw)
                    
                    print(f"  [Replacement] Replacing AveragePooling2D '{name}' (pool={pool_size}) with DepthwiseConv2D...")
                    new_layer = layers.DepthwiseConv2D(
                        kernel_size=pool_size,
                        strides=(1, 1),
                        padding=padding,
                        depth_multiplier=1,
                        use_bias=False,
                        depthwise_initializer=tf.keras.initializers.Constant(scale),
                        trainable=False,
                        name=name
                    )
                    x = new_layer(layer_inputs)
                else:
                    # Apply original layer to new inputs
                    x = layer(layer_inputs)
                    
                outputs = layer.output
                if isinstance(outputs, list):
                    for orig_out, new_out in zip(outputs, x):
                        tensor_map[id(orig_out)] = new_out
                else:
                    tensor_map[id(outputs)] = x
                    
            new_outputs = [tensor_map[id(out)] for out in model.outputs]
            model = models.Model(inputs=new_inputs, outputs=new_outputs, name=model.name + "_optimized")
            print("[+] Model graph reconstruction completed successfully.")
        except Exception as reconstruct_err:
            print(f"[WARNING] Failed to reconstruct graph: {reconstruct_err}. Proceeding with original layers.")

    # 2. Trim outputs if requested
    if trim_outputs:
        raw_heatmap_output = model.get_layer("predicted_heatmap_raw").output
        print(f"[Optimizer] Trimming model outputs: keeping only raw heatmap '{raw_heatmap_output.name}'")
        model = tf.keras.models.Model(inputs=model.inputs, outputs=[raw_heatmap_output], name=model.name + "_trimmed")
    return model


def main():
    parser = argparse.ArgumentParser(description="Statically convert Tracker Ver 4 Keras model to TFLite.")
    parser.add_argument(
        "--keras_in",
        default="outputs/tracker.keras",
        help="Path to input Keras model weights/file (.keras)"
    )
    parser.add_argument(
        "--tflite_out",
        default="outputs/tracker.tflite",
        help="Path to save the compiled static TFLite model (.tflite)"
    )
    parser.add_argument(
        "--copy_to_android",
        action="store_true",
        help="Copy the compiled TFLite model directly to Android assets"
    )
    parser.add_argument(
        "--quant",
        default="none",
        choices=["none", "fp16", "int8", "int8_io"],
        help="Quantization type: none (FP32), fp16 (float16), int8 (integer weights/activations), int8_io (integer inputs/outputs)"
    )
    parser.add_argument(
        "--h5_dataset",
        default="dataset_generator/compiled/dataset.h5",
        help="Path to HDF5 dataset.h5 file to use as calibration data for INT8 quantization"
    )
    parser.add_argument(
        "--qat",
        action="store_true",
        help="Enable QAT preservation mode: converts the loaded QAT model directly to retain activation scale factors instead of reconstructing"
    )
    
    parser.add_argument(
        "--input_tensors",
        default=None,
        help="Comma-separated list of input tensor names for subgraph extraction"
    )
    parser.add_argument(
        "--output_tensors",
        default=None,
        help="Comma-separated list of output tensor names for subgraph extraction"
    )
    
    args = parser.parse_args()
    
    # 1. Validate inputs
    if not os.path.exists(args.keras_in):
        print(f"Error: Keras model file '{args.keras_in}' not found.", file=sys.stderr)
        sys.exit(1)
        
    print("[*] Importing TensorFlow...")
    try:
        import tensorflow as tf
    except ImportError:
        print("Error: TensorFlow is not installed in this Python environment.", file=sys.stderr)
        sys.exit(1)
        
    # Monkeypatch tf.transpose to replace layout-preserving transposes with reshapes during concrete function tracing
    orig_transpose = tf.transpose
    def custom_transpose(a, perm=None, conjugate=False, name=None):
        if perm is not None:
            try:
                perm_list = [int(p) for p in perm]
                shape = a.shape.as_list() if hasattr(a.shape, 'as_list') else list(a.shape)
                if transpose_preserves_memory_layout(shape, perm_list):
                    target_shape = [shape[i] for i in perm_list]
                    cleaned_target_shape = [(-1 if dim is None else int(dim)) for dim in target_shape]
                    print(f"[Optimizer] Intercepted tf.transpose. Replacing shape {shape} (perm {perm_list}) with tf.reshape to {cleaned_target_shape}")
                    return tf.reshape(a, cleaned_target_shape, name=name)
            except Exception as e:
                print(f"[Optimizer] Warning in custom_transpose intercept: {e}")
        return orig_transpose(a, perm=perm, conjugate=conjugate, name=name)
    tf.transpose = custom_transpose
        
    # Append current dir and parent dir to sys.path to load tracker_model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    sys.path.append(os.path.dirname(script_dir))
    
    try:
        from tracker_model import TargetTrackerVer4
        import tracker_model
    except ImportError:
        print("Error: Could not import TargetTrackerVer4 from tracker_model.py. Ensure this script is run from the tracker_ver4 folder.", file=sys.stderr)
        sys.exit(1)

    # 2. Reconstruct Model Architecture
    print("[*] Reconstructing model architecture...")
    try:
        # Load the trained model to inspect its architecture
        print(f"[*] Inspecting trained model structure from: {args.keras_in} ...")
        custom_objects = {
            "DepthwiseCorrelationFusion": tracker_model.DepthwiseCorrelationFusion,
            "DepthToSpace": tracker_model.DepthToSpace,
        }
        if args.qat or "qat" in os.path.basename(args.keras_in).lower():
            import tensorflow_model_optimization as tfmot
            
            @tf.keras.utils.register_keras_serializable(package="Custom")
            class CustomLayerQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
                def get_weights_and_quantizers(self, layer):
                    return []
                def get_activations_and_quantizers(self, layer):
                    return []
                def set_quantize_weights(self, layer, quantize_weights):
                    pass
                def set_quantize_activations(self, layer, quantize_activations):
                    pass
                def get_output_quantizers(self, layer):
                    return [
                        tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
                            num_bits=8, per_axis=False, symmetric=False, narrow_range=False
                        )
                    ]
                def get_config(self):
                    return {}
            
            custom_objects["CustomLayerQuantizeConfig"] = CustomLayerQuantizeConfig
            
            # Monkeypatch QuantizeWrapperV2.build to build the inner layer first.
            # This works around a tfmot initialization order bug when deserializing a QAT functional model
            # where self.layer.trainable_weights is empty during build() because the inner layer wasn't built yet.
            orig_build = tfmot.quantization.keras.QuantizeWrapperV2.build
            def custom_build(self, input_shape):
                if not self.layer.built:
                    self.layer.build(input_shape)
                orig_build(self, input_shape)
            tfmot.quantization.keras.QuantizeWrapperV2.build = custom_build
            
            with tfmot.quantization.keras.quantize_scope(custom_objects):
                loaded_model = tf.keras.models.load_model(args.keras_in, compile=False, safe_mode=False)
        else:
            loaded_model = tf.keras.models.load_model(args.keras_in, compile=False, safe_mode=False, custom_objects=custom_objects)
        
        # Check model type
        is_pixel_model = False
        print("[+] Loaded model type: TargetTrackerVer4")

        # Detect parameters
        detected_config = {}
        has_layer = lambda name: any(name in l.name for l in loaded_model.layers)
        
        # A. Attention
        if has_layer("depthwise_correlation_fusion"):
            detected_config["attention_mechanism"] = "depthwise_corr"
        elif has_layer("mha_fusion"):
            detected_config["attention_mechanism"] = "multi_head_cross"
        elif has_layer("attention_dot"):
            detected_config["attention_mechanism"] = "dot_cross"
        else:
            detected_config["attention_mechanism"] = "linear_cross"

        # B. Decoder
        if any("ps_shuffle" in l.name or "pixel_shuffle" in l.name for l in loaded_model.layers):
            detected_config["decoder_type"] = "pixel_shuffle"
        elif has_layer("decoder_skip1_proj") or has_layer("decoder_skip2_proj"):
            detected_config["decoder_type"] = "fpn_add"
        elif has_layer("decoder_skip1") or any("concat" in l.name for l in loaded_model.layers if "decoder" in l.name):
            detected_config["decoder_type"] = "unet"
        else:
            detected_config["decoder_type"] = "light_naive"

        # C. Search/Shared backbone & Width multiplier
        try:
            if is_pixel_model:
                search_extractor = loaded_model.get_layer("shared_siamese_backbone")
            else:
                search_extractor = loaded_model.get_layer("search_feature_extractor")
            sb_layer_names = [l.name for l in search_extractor.layers]
            
            backbone_key = "shared_backbone" if is_pixel_model else "search_backbone"
            
            if any("sb_alex_conv" in name for name in sb_layer_names):
                detected_config[backbone_key] = "alex_net"
                sb_final_filters = search_extractor.get_layer("sb_alex_conv5").filters
                detected_config["width_multiplier"] = sb_final_filters / 128.0
            elif any("sb_dw1" in name for name in sb_layer_names):
                detected_config[backbone_key] = "mnv1"
                detected_config["width_multiplier"] = 1.0
            elif any("sb_ir1_conv" in name for name in sb_layer_names):
                detected_config[backbone_key] = "yolo5"
                detected_config["width_multiplier"] = 1.0
            elif any("sb_ir1" in name for name in sb_layer_names):
                sb_final_filters = search_extractor.get_layer("sb_final_conv").filters
                detected_config["width_multiplier"] = sb_final_filters / 128.0
                
                sb_ir3_layer = [l for l in search_extractor.layers if "sb_ir3_project" in l.name]
                if sb_ir3_layer:
                    ratio = sb_ir3_layer[0].filters / detected_config["width_multiplier"]
                    if ratio <= 40.0:
                        detected_config[backbone_key] = "mnv2_nano"
                    else:
                        detected_config[backbone_key] = "mnv2"
                else:
                    detected_config[backbone_key] = "mnv2_nano"
            else:
                detected_config[backbone_key] = "custom_legacy"
                detected_config["width_multiplier"] = 1.0
        except Exception as e:
            print(f"Warning: Failed to auto-detect search/shared backbone. Defaulting. Details: {e}")
            backbone_key = "shared_backbone" if is_pixel_model else "search_backbone"
            detected_config[backbone_key] = "mnv2_nano"
            detected_config["width_multiplier"] = 0.5

        # D. Reference backbone (Only for TargetTrackerVer4)
        if not is_pixel_model:
            try:
                ref_encoder = loaded_model.get_layer("reference_target_encoder")
                ref_layer_names = [l.name for l in ref_encoder.layers]
                
                if any("ref_alex_conv" in name for name in ref_layer_names):
                    detected_config["reference_backbone"] = "alex_net"
                elif any("ref_dw1" in name for name in ref_layer_names):
                    detected_config["reference_backbone"] = "mnv1"
                elif any("ref_ir1" in name for name in ref_layer_names):
                    # Distinguish custom_legacy: check if ref_init_conv has larger filters
                    ref_init_layer = [l for l in ref_encoder.layers if "ref_init_conv" in l.name]
                    if ref_init_layer and (ref_init_layer[0].filters / detected_config["width_multiplier"] > 24.0):
                        detected_config["reference_backbone"] = "custom_legacy"
                    elif any("ref_ir2" in name for name in ref_layer_names):
                        detected_config["reference_backbone"] = "mini_mnv2"
                    else:
                        detected_config["reference_backbone"] = "mnv2"
                else:
                    detected_config["reference_backbone"] = "custom_legacy"
            except Exception as e:
                print(f"Warning: Failed to auto-detect reference backbone. Defaulting. Details: {e}")
                detected_config["reference_backbone"] = "mini_mnv2"
        # E. Normalization Type
        all_layer_types = []
        all_layer_names = []
        
        def collect_layers(m):
            for l in m.layers:
                all_layer_types.append(type(l).__name__)
                all_layer_names.append(l.name)
                # Check for nested sub-models/layers
                if hasattr(l, "layers"):
                    collect_layers(l)
                    
        collect_layers(loaded_model)
        
        has_bn = any("BatchNormalization" in t for t in all_layer_types)
        has_gn = any("GroupNormalization" in t for t in all_layer_types)
        
        if not has_bn and not has_gn:
            # Check for folded activation layers like sb_init_gn, quality_gn1
            has_folded = any("_gn" in name or "_bn" in name for name in all_layer_names)
            if has_folded:
                detected_config["normalization_type"] = "folded"
            else:
                detected_config["normalization_type"] = "group_norm"
        elif has_bn:
            detected_config["normalization_type"] = "batch_norm"
        else:
            detected_config["normalization_type"] = "group_norm"
        print("[+] Auto-detected architecture parameters:")
        for k, v in detected_config.items():
            print(f"    - {k}: {v}")

        # Instantiate tracker and override its config with the detected parameters
        if is_pixel_model:
            if TargetTrackerVerPixel is None:
                raise ValueError("TargetTrackerVerPixel cannot be imported from tracker_ver4_pxl.py")
            tracker = TargetTrackerVerPixel()
        else:
            tracker = TargetTrackerVer4()
            
        tracker.config.update(detected_config)
        
        # Create model structure
        model = tracker.create_model()
        print("[+] Model architecture successfully built.")
    except Exception as e:
        print(f"Error: Failed to build model architecture. Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 3. Choose conversion target model
    if args.qat or "qat" in os.path.basename(args.keras_in).lower():
        print("[*] QAT mode detected or enabled. Converting the loaded model directly to preserve learned scale factors...")
        conversion_model = loaded_model
    else:
        # Reconstruct and copy weights
        print(f"[*] Copying weights layer-by-layer from: {args.keras_in} ...")
        try:
            for layer in model.layers:
                # Skip layers that do not contain any weights (e.g. TFOpLambda, Input, Reshape, Transpose)
                if len(layer.weights) == 0:
                    continue
                loaded_layer = loaded_model.get_layer(layer.name)
                if len(loaded_layer.weights) != len(layer.weights):
                    print(f"[WARNING] Weight count mismatch for layer '{layer.name}': target has {len(layer.weights)} weights, source has {len(loaded_layer.weights)}. Retaining initialized weights.")
                    continue
                layer.set_weights(loaded_layer.get_weights())
            print("[+] Weights loaded successfully via layer-by-layer copy.")
        except Exception as e:
            print(f"Error: Failed to load weights. Ensure the Keras file contains compatible weights.\nDetails: {e}", file=sys.stderr)
            sys.exit(1)
        conversion_model = model
        
    # Save original inputs and outputs for potential calibration/mapping
    original_inputs_for_calib = conversion_model.inputs
    original_outputs_for_calib = conversion_model.outputs
    target_inputs_for_calib = None

    # Extract subgraph if requested
    if args.input_tensors and args.output_tensors:
        input_names = [name.strip() for name in args.input_tensors.split(",") if name.strip()]
        output_names = [name.strip() for name in args.output_tensors.split(",") if name.strip()]
        print(f"[*] Extracting subgraph for inputs={input_names} and outputs={output_names}...")
        try:
            conversion_model = optimize_and_trim_model(conversion_model, trim_outputs=False)
            conversion_model, target_inputs_for_calib = build_subgraph(conversion_model, input_names, output_names)
            print("[+] Subgraph extraction completed successfully.")
        except Exception as e:
            print(f"Error: Subgraph extraction failed. Details: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Optimize conversion model (replace memory-layout-preserving transposes with reshapes)
        print("[*] Performing functional graph surgery on conversion model...")
        try:
            conversion_model = optimize_and_trim_model(conversion_model, trim_outputs=True)
            print("[+] Graph surgery completed successfully.")
        except Exception as surgery_err:
            print(f"[WARNING] Graph surgery failed: {surgery_err}. Proceeding with original model.")
        
    # 4. Generate Concrete Function with Static Batch Shape (batch_size = 1)
    print("[*] Defining concrete function with static input shapes (forcing batch_size=1)...")
    try:
        input_specs = []
        for inp in conversion_model.inputs:
            shape = inp.shape.as_list()
            if len(shape) > 0 and (shape[0] is None or shape[0] == -1):
                shape[0] = 1
            spec = tf.TensorSpec(shape, dtype=inp.dtype, name=inp.name.split(":")[0])
            input_specs.append(spec)
            print(f"[Concrete Function] Input spec: {spec}")

        @tf.function
        def run_model(*inputs):
            return conversion_model(inputs)
            
        concrete_func = run_model.get_concrete_function(*input_specs)
        print("[+] Static concrete function generated.")
    except Exception as e:
        print(f"Error: Failed to generate concrete function. Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 5. Convert to TFLite
    print("[*] Converting model to static TFLite format (this may take a moment)...")
    try:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        
        # Configure quantization options
        if args.quant == "fp16":
            print("[*] Applying Float16 quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif args.quant in ["int8", "int8_io"]:
            print(f"[*] Applying INT8 quantization (mode: {args.quant})...")
            
            # Define representative dataset generator for integer calibration
            def representative_dataset_gen():
                import h5py
                h5_path = args.h5_dataset
                if not os.path.isabs(h5_path):
                    if os.path.exists(h5_path):
                        h5_path = os.path.abspath(h5_path)
                    else:
                        h5_path = os.path.join(os.path.dirname(script_dir), h5_path)
                
                if not os.path.exists(h5_path):
                    raise FileNotFoundError(f"HDF5 dataset for calibration not found at: {h5_path}")
                
                if target_inputs_for_calib is not None:
                    # Construct a mapping model to compute intermediate tensors for calibration
                    calib_mapper = tf.keras.models.Model(
                        inputs=original_inputs_for_calib,
                        outputs=target_inputs_for_calib
                    )
                else:
                    calib_mapper = None
                    
                with h5py.File(h5_path, 'r') as f:
                    ref_ds = f['reference_stack']
                    search_ds = f['search_frame']
                    total_samples = ref_ds.shape[0]
                    
                    # Calibrate on up to 100 representative samples
                    num_calibration = min(100, total_samples)
                    indices = np.linspace(0, total_samples - 1, num_calibration, dtype=int)
                    
                    for idx in indices:
                        ref_val = np.expand_dims(ref_ds[idx], axis=0).astype(np.float32)
                        search_val = np.expand_dims(search_ds[idx], axis=0).astype(np.float32)
                        
                        if calib_mapper is not None:
                            mapped_vals = calib_mapper([ref_val, search_val])
                            if not isinstance(mapped_vals, list):
                                mapped_vals = [mapped_vals]
                            yield [val.numpy() if hasattr(val, "numpy") else val for val in mapped_vals]
                        else:
                            yield [ref_val, search_val]
            
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            
            if args.quant == "int8_io":
                print("[*] Enforcing integer input/output tensors...")
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                
        tflite_model = converter.convert()
        print("[+] Model converted to TFLite successfully.")
    except Exception as e:
        print(f"Error: TFLite conversion failed. Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 6. Save compiled model
    out_dir = os.path.dirname(args.tflite_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    try:
        with open(args.tflite_out, "wb") as f:
            f.write(tflite_model)
        print(f"[SUCCESS] Static TFLite model saved at: {args.tflite_out}")
    except Exception as e:
        print(f"Error: Failed to save TFLite model to '{args.tflite_out}'. Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 7. Optionally copy to Android assets
    if args.copy_to_android:
        android_assets_dir = os.path.join(os.path.dirname(script_dir), "android", "app", "src", "main", "assets")
        if os.path.exists(android_assets_dir):
            android_tflite_path = os.path.join(android_assets_dir, "tracker.tflite")
            try:
                shutil.copy2(args.tflite_out, android_tflite_path)
                print(f"[SUCCESS] Copied compiled model directly to Android assets: {android_tflite_path}")
            except Exception as e:
                print(f"Warning: Failed to copy model to Android assets. Details: {e}", file=sys.stderr)
        else:
            print(f"Warning: Android assets folder not found at: {android_assets_dir}. Skipping copy.", file=sys.stderr)

if __name__ == "__main__":
    main()
