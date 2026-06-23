import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import os
import pickle
import numpy as np
import configparser
import h5py
import psutil
import json


def configure_tensorflow_gpu_memory():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return

    limit_raw = os.environ.get("TRACKER_GPU_MEMORY_LIMIT_MB", "").strip()
    if limit_raw:
        try:
            limit_mb = float(limit_raw)
        except ValueError:
            print(f"[TF CONFIG] Ignoring invalid TRACKER_GPU_MEMORY_LIMIT_MB={limit_raw!r}")
            limit_mb = 0.0

        if limit_mb > 0.0:
            try:
                for gpu in gpus:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)]
                    )
                print(f"[TF CONFIG] Limited each GPU logical device to {limit_mb:.0f} MB.")
                return
            except RuntimeError as exc:
                print(f"[TF CONFIG] Could not set GPU memory limit after runtime initialization: {exc}")

    allow_growth = os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").strip().lower()
    if allow_growth in ("1", "true", "yes", "on"):
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[TF CONFIG] Enabled GPU memory growth.")
        except RuntimeError as exc:
            print(f"[TF CONFIG] Could not enable GPU memory growth after runtime initialization: {exc}")


configure_tensorflow_gpu_memory()

# Global Normalization setting for dynamic layer configuration
_NORMALIZATION_TYPE = "group_norm"
_OVERRIDE_USE_BIAS = False

_orig_conv2d = layers.Conv2D
_orig_dw_conv2d = layers.DepthwiseConv2D
_orig_conv2d_transpose = layers.Conv2DTranspose

def _smart_conv2d(*args, **kwargs):
    global _NORMALIZATION_TYPE, _OVERRIDE_USE_BIAS
    if _OVERRIDE_USE_BIAS and _NORMALIZATION_TYPE in ["batch_norm", "folded"]:
        kwargs["use_bias"] = True
    return _orig_conv2d(*args, **kwargs)

def _smart_dw_conv2d(*args, **kwargs):
    global _NORMALIZATION_TYPE, _OVERRIDE_USE_BIAS
    if _OVERRIDE_USE_BIAS and _NORMALIZATION_TYPE in ["batch_norm", "folded"]:
        kwargs["use_bias"] = True
    return _orig_dw_conv2d(*args, **kwargs)

def _smart_conv2d_transpose(*args, **kwargs):
    global _NORMALIZATION_TYPE, _OVERRIDE_USE_BIAS
    if _OVERRIDE_USE_BIAS and _NORMALIZATION_TYPE in ["batch_norm", "folded"]:
        kwargs["use_bias"] = True
    return _orig_conv2d_transpose(*args, **kwargs)

layers.Conv2D = _smart_conv2d
layers.DepthwiseConv2D = _smart_dw_conv2d
layers.Conv2DTranspose = _smart_conv2d_transpose

# Save the original GroupNormalization
if hasattr(layers, 'GroupNormalization'):
    GroupNormClass = layers.GroupNormalization
else:
    try:
        import tensorflow_addons as tfa
        GroupNormClass = tfa.layers.GroupNormalization
    except ImportError:
        GroupNormClass = None

def get_safe_groups(channels, requested_groups=8):
    if channels is None or requested_groups is None:
        return 8
    if channels % requested_groups == 0:
        return requested_groups
    for candidate in range(requested_groups, 0, -1):
        if channels % candidate == 0:
            return candidate
    return 1

def _GroupNormalization(channels, name=None, **kwargs):
    global _NORMALIZATION_TYPE
    if _NORMALIZATION_TYPE == "batch_norm":
        return layers.BatchNormalization(name=name, **kwargs)
    elif _NORMALIZATION_TYPE == "folded":
        return layers.Activation("linear", name=name, **kwargs)
    else:
        g = get_safe_groups(channels)
        if GroupNormClass is not None:
            return GroupNormClass(groups=g, name=name, **kwargs)
        else:
            return layers.Layer(name=name, **kwargs)

@tf.keras.utils.register_keras_serializable(package="Custom")
class DepthwiseCorrelationFusion(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        search_feat, ref_feat = inputs
        batch_size = search_feat.shape[0]
        if batch_size is None:
            batch_size = tf.shape(search_feat)[0]
        h_s, w_s, c = search_feat.shape[1], search_feat.shape[2], search_feat.shape[3]
        h_r, w_r = ref_feat.shape[1], ref_feat.shape[2]
        
        # Group batch elements along the channel dimension
        s_transposed = tf.transpose(search_feat, [1, 2, 0, 3])
        s_reshaped = tf.reshape(s_transposed, [1, h_s, w_s, batch_size * c])
        
        r_transposed = tf.transpose(ref_feat, [1, 2, 0, 3])
        r_filter = tf.reshape(r_transposed, [h_r, w_r, batch_size * c, 1])
        
        out = tf.nn.depthwise_conv2d(s_reshaped, r_filter, strides=[1, 1, 1, 1], padding="SAME")
        
        out_reshaped = tf.reshape(out, [h_s, w_s, batch_size, c])
        return tf.transpose(out_reshaped, [2, 0, 1, 3])

@tf.keras.utils.register_keras_serializable(package="Custom")
class Conv2DCorrelationFusion(layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(1, 1, 1, self.channels),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel"
        )
        self.bias = self.add_weight(
            shape=(self.channels,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )
        super().build(input_shape)
        
    def call(self, inputs):
        search_feat, ref_feat = inputs
        batch_size = search_feat.shape[0]
        if batch_size is None:
            batch_size = tf.shape(search_feat)[0]
        h_s, w_s, c = search_feat.shape[1], search_feat.shape[2], search_feat.shape[3]
        h_r, w_r = ref_feat.shape[1], ref_feat.shape[2]
        
        # 1. Depthwise correlation
        s_transposed = tf.transpose(search_feat, [1, 2, 0, 3])
        s_reshaped = tf.reshape(s_transposed, [1, h_s, w_s, batch_size * c])
        
        r_transposed = tf.transpose(ref_feat, [1, 2, 0, 3])
        r_filter = tf.reshape(r_transposed, [h_r, w_r, batch_size * c, 1])
        
        out = tf.nn.depthwise_conv2d(s_reshaped, r_filter, strides=[1, 1, 1, 1], padding="SAME")
        out_reshaped = tf.reshape(out, [h_s, w_s, batch_size, c])
        dw_corr = tf.transpose(out_reshaped, [2, 0, 1, 3])
        
        # 2. Sum across channels to produce 1-channel similarity map
        similarity = tf.reduce_sum(dw_corr, axis=-1, keepdims=True)
        
        # 3. Project to channels using 1x1 conv formula
        proj = tf.nn.conv2d(similarity, self.kernel, strides=[1, 1, 1, 1], padding="SAME")
        return tf.nn.bias_add(proj, self.bias)
        
    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config

@tf.keras.utils.register_keras_serializable(package="Custom")
class DepthToSpace(layers.Layer):
    def __init__(self, block_size=2, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        
    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.block_size)
        
    def get_config(self):
        config = super().get_config()
        config.update({"block_size": self.block_size})
        return config

def load_model_config(config_path="model.conf"):
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        alt_path = os.path.join(os.path.dirname(__file__), config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
    config = configparser.ConfigParser()
    config.read(config_path)
    
    ref_backbone = config.get("Backbone", "reference_backbone", fallback="mini_mnv2")
    search_backbone = config.get("Backbone", "search_backbone", fallback="mnv2_nano")
    width_mult = config.getfloat("Backbone", "width_multiplier", fallback=0.5)
    
    attn_mech = config.get("Attention", "mechanism", fallback="depthwise_corr")
    
    dec_type = config.get("Decoder", "type", fallback="fpn_add")
    heatmap_pxl_size = config.getint("Decoder", "heatmap_pxl_size", fallback=64)
    heatmap_output_source = config.get("Decoder", "heatmap_output_source", fallback="pre_threshold")
    
    hm_loss_default = config.get("Loss", "heatmap_loss_default", fallback="dbsz_relu")
    soft_argmax_beta = config.getfloat("Loss", "soft_argmax_beta", fallback=15.0)
    
    norm_type = config.get("Normalization", "type", fallback="group_norm")

    stack_layers      = config.getint("Stack", "stack_layers",      fallback=16)
    stack_target_size = config.getint("Stack", "stack_target_size", fallback=64)
    reference_feature_size = config.getint("Stack", "reference_feature_size", fallback=8)
    reference_downscale = config.get("Stack", "reference_downscale", fallback="auto")

    return {
        "reference_backbone": ref_backbone,
        "search_backbone": search_backbone,
        "width_multiplier": width_mult,
        "attention_mechanism": attn_mech,
        "decoder_type": dec_type,
        "heatmap_pxl_size": heatmap_pxl_size,
        "heatmap_output_source": heatmap_output_source,
        "heatmap_loss_default": hm_loss_default,
        "soft_argmax_beta": soft_argmax_beta,
        "normalization_type": norm_type,
        "stack_layers": stack_layers,
        "stack_target_size": stack_target_size,
        "reference_feature_size": reference_feature_size,
        "reference_downscale": reference_downscale,
    }

def check_and_create_default_config(config_path="model.conf"):
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        alt_path = os.path.join(os.path.dirname(__file__), config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            config_path = alt_path
    if not os.path.exists(config_path):
        default_content = """# TargetTrackerVer5 Model Configuration File
# This file configures the backbones, attention mechanism, and decoder of the tracking network.
# Optimizing these parameters helps balance tracking precision and real-time execution on weak boards.

[Backbone]
# Reference Stack Backbone
# Options:
#   - alex_dense      : AlexNet-like dense Conv stem with larger early receptive fields.
#   - mini_mnv2       : (Recommended) Compact MobileNetV2 with low channel widths (max 64), customized for 32x32 inputs.
#   - mnv1            : Standard MobileNetV1 backbone.
#   - mnv2            : Full MobileNetV2 backbone.
#   - yolo5           : CSPDarknet-style backbone.
#   - unet            : UNet-style encoder, 2 Conv blocks per scale, 3 downsampling stages (stride x8).
#   - custom_legacy   : The original hardcoded tracker_ver5 reference backbone.
reference_backbone = mini_mnv2

# Search Frame Backbone
# Options:
#   - alex_dense      : AlexNet-like dense Conv stem with larger early receptive fields.
#   - mnv2_nano       : (Recommended) Highly optimized MobileNetV2 with capped channel widths (max 64) for low latency.
#   - mnv1            : Standard MobileNetV1 backbone.
#   - mnv2            : Full MobileNetV2 backbone.
#   - yolo5           : CSPDarknet-style backbone.
#   - unet            : UNet-style encoder, 2 Conv blocks per scale, 4 downsampling stages (stride x16).
#   - custom_legacy   : The original hardcoded tracker_ver5 search backbone.
search_backbone = mnv2_nano

# Channel width multiplier to scale both backbones (e.g., 0.5, 0.75, 1.0). Smaller values reduce FLOPs.
width_multiplier = 0.5

[Attention]
# Fusion Attention Mechanism
# Options:
#   - depthwise_corr  : (Recommended) Depthwise Cross-Correlation. 0 learnable parameters, extremely fast on CPUs.
#   - dot_cross       : Standard single-head dot-product cross-attention.
#   - linear_cross    : Linearized cross-attention. Reduces memory complexity from O(N^2) to O(N).
#   - multi_head_cross: Multi-head cross-attention. High capacity, slightly more resource intensive.
mechanism = depthwise_corr

[Decoder]
# Heatmap Decoder Architecture
# Options:
#   - fpn_add         : (Recommended) Feature Pyramid Network decoder using skip-add connections. Saves RAM bandwidth.
#   - unet            : Standard U-Net style decoder with skip-concatenations.
#   - pixel_shuffle   : Uses sub-pixel convolutions (depth-to-space). Fast on NPUs.
#   - light_naive     : Transposed convolutions without skip connections. Fastest, lower precision.
type = fpn_add

# Target resolution of the output heatmap (e.g. 16, 32, 64, 128, 256)
heatmap_pxl_size = 64

# Main model heatmap output used by training loss.
# Options: raw, norm, pre_threshold, processed
# processed keeps the legacy threshold+blur output; pre_threshold trains before ReLU+blur.
heatmap_output_source = pre_threshold

[Loss]
# Default loss function for heatmap regression
# Options: dbsz_relu, dbsz_soft, dbsz_hard, centernet_dice, focal_dice, adaptive_wing, mse
heatmap_loss_default = dbsz_relu

# Soft-argmax temperature for coordinate loss. Lower values give smoother gradients early in training.
soft_argmax_beta = 15.0

[Normalization]
# Normalization Type
# Options: group_norm, batch_norm
type = group_norm

[Stack]
# Number of multi-scale layers in the reference stack.
# Must match compiler.stack_layers in dataset_generator/pipeline_config.json.
stack_layers = 16

# Spatial size (px) of each reference stack frame.
# Must match compiler.stack_target_size in dataset_generator/pipeline_config.json.
stack_target_size = 64

# Spatial size of reference encoder features used as depthwise-correlation weights.
# Keep smaller than the search feature map, usually 8 when search bottleneck is 16.
reference_feature_size = 8

# Options: auto, conv, depthwise, avg_pool, max_pool
reference_downscale = auto
"""
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(default_content)
        print(f"\n[CONFIG NOTICE] Configuration file '{config_path}' was not found.")
        print(f"Created a default '{config_path}' with optimal lightweight parameters.")
        print("Please check the configuration options, modify if needed, and run the training/execution again.\n")
        import sys
        sys.exit(0)

# =====================================================================
# Coordinate-Based Loss Helpers
# =====================================================================

def get_peak_coords_tf(heatmap):
    H = tf.shape(heatmap)[1]
    W = tf.shape(heatmap)[2]
    # Flatten spatial dimensions to (B, H*W)
    flat_hm = tf.reshape(heatmap, [-1, H * W])
    flat_idx = tf.argmax(flat_hm, axis=-1)
    
    # Convert flat indices back to y, x coordinates
    W_64 = tf.cast(W, tf.int64)
    y = tf.cast(flat_idx // W_64, tf.float32)
    x = tf.cast(flat_idx % W_64, tf.float32)
    return tf.stack([y, x], axis=-1)

def soft_argmax_2d(heatmap, beta=30.0):
    # heatmap has shape (B, H, W, 1)
    B = tf.shape(heatmap)[0]
    H = tf.shape(heatmap)[1]
    W = tf.shape(heatmap)[2]
    
    # Flatten spatial dimensions
    flat_hm = tf.reshape(heatmap, [B, H * W]) # shape (B, H*W)
    
    # Apply softmax with temperature beta
    probs = tf.nn.softmax(beta * flat_hm, axis=-1) # shape (B, H*W)
    
    # Create coordinates grid normalized to [0.0, 1.0] range
    y_grid, x_grid = tf.meshgrid(
        tf.range(H, dtype=tf.float32) / tf.cast(H, tf.float32),
        tf.range(W, dtype=tf.float32) / tf.cast(W, tf.float32),
        indexing='ij'
    )
    
    # Flatten grids
    flat_y = tf.reshape(y_grid, [-1]) # shape (H*W,)
    flat_x = tf.reshape(x_grid, [-1]) # shape (H*W,)
    
    # Compute expected values
    pred_y = tf.reduce_sum(probs * flat_y, axis=-1, keepdims=True) # shape (B, 1)
    pred_x = tf.reduce_sum(probs * flat_x, axis=-1, keepdims=True) # shape (B, 1)
    
    return tf.concat([pred_y, pred_x], axis=-1) # shape (B, 2)

def coordinate_distance_loss(gt_coords, pred_heatmap, gt_quality, c_bg=3.0, beta=15.0):
    # 1. Differentiable Soft-Argmax to extract coordinates from predicted heatmap
    pred_coords_soft = soft_argmax_2d(pred_heatmap, beta=beta)
    
    # 2. Scale both predicted and ground-truth coordinates to 256.0 space to compute Huber loss in pixel units
    huber = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)
    loss_coords = huber(gt_coords * 256.0, pred_coords_soft * 256.0)
    loss_coords = tf.reshape(loss_coords, [-1, 1])
    
    # 3. DBSZ ReLU / MSE loss on negative samples (predicted heatmap vs zero map)
    # When gt_true is all zeros, dbsz_relu simplifies exactly to: c_bg * Mean(y_pred^2)
    loss_neg_heatmap = c_bg * tf.reduce_mean(tf.square(pred_heatmap), axis=[1, 2, 3])
    loss_neg_heatmap = tf.reshape(loss_neg_heatmap, [-1, 1])
    
    # 4. Create masks based on target quality threshold (0.5)
    pos_mask = tf.cast(gt_quality > 0.5, tf.float32)
    neg_mask = 1.0 - pos_mask
    
    # Normalize losses by the count of positive and negative samples in the batch respectively
    pos_count = tf.reduce_sum(pos_mask)
    neg_count = tf.reduce_sum(neg_mask)
    
    # Calculate normalized positive and negative losses
    loss_pos_normalized = tf.reduce_sum(loss_coords * pos_mask) / (pos_count + 1e-7)
    loss_neg_normalized = tf.reduce_sum(loss_neg_heatmap * neg_mask) / (neg_count + 1e-7)
    
    # Apply conditional mask to avoid adding values if there are no samples of a class in the batch
    loss_pos_final = tf.where(pos_count > 0.5, loss_pos_normalized, 0.0)
    loss_neg_final = tf.where(neg_count > 0.5, loss_neg_normalized, 0.0)
    
    return loss_pos_final + loss_neg_final

# =====================================================================
# Batch Normalization Folding Helper
# =====================================================================

def get_all_leaf_layers(model):
    leaf_layers = []
    for layer in model.layers:
        if hasattr(layer, "layers"):
            leaf_layers.extend(get_all_leaf_layers(layer))
        else:
            leaf_layers.append(layer)
    return leaf_layers

def fold_model_weights(unfolded_model, folded_model):
    unfolded_leafs = get_all_leaf_layers(unfolded_model)
    folded_leafs = get_all_leaf_layers(folded_model)
    
    folded_layers = {layer.name: layer for layer in folded_leafs}
    folded_bn_names = set()
    folded_conv_names = set()
    
    for bn_layer in unfolded_leafs:
        if bn_layer.__class__.__name__ == 'BatchNormalization':
            try:
                if hasattr(bn_layer.input, '_keras_history'):
                    parent_layer, _, _ = bn_layer.input._keras_history
                else:
                    parent_layer, _, _ = bn_layer.input.keras_history
            except Exception:
                continue
                
            if parent_layer.__class__.__name__ in ('Conv2D', 'DepthwiseConv2D', 'Conv2DTranspose'):
                bn_weights = bn_layer.get_weights()
                if not bn_weights or len(bn_weights) < 4:
                    continue
                gamma, beta, mean, var = bn_weights
                epsilon = bn_layer.epsilon
                
                conv_weights = parent_layer.get_weights()
                if len(conv_weights) == 2:
                    kernel, bias = conv_weights
                elif len(conv_weights) == 1:
                    kernel = conv_weights[0]
                    if parent_layer.__class__.__name__ == 'Conv2DTranspose':
                        bias = np.zeros((kernel.shape[-2],), dtype=np.float32)
                    else:
                        bias = np.zeros((kernel.shape[-1],), dtype=np.float32)
                else:
                    continue
                
                scale = gamma / np.sqrt(var + epsilon)
                
                if parent_layer.__class__.__name__ == 'DepthwiseConv2D':
                    depth_multiplier = kernel.shape[-1]
                    c_in = kernel.shape[-2]
                    scale_reshaped = np.reshape(scale, (1, 1, c_in, depth_multiplier))
                    kernel_folded = kernel * scale_reshaped
                elif parent_layer.__class__.__name__ == 'Conv2DTranspose':
                    scale_reshaped = np.reshape(scale, (1, 1, -1, 1))
                    kernel_folded = kernel * scale_reshaped
                else:
                    scale_reshaped = np.reshape(scale, (1, 1, 1, -1))
                    kernel_folded = kernel * scale_reshaped
                    
                bias_folded = (bias - mean) * scale + beta
                
                folded_conv_layer = folded_layers.get(parent_layer.name)
                if folded_conv_layer is not None:
                    folded_conv_layer.set_weights([kernel_folded, bias_folded])
                    folded_bn_names.add(bn_layer.name)
                    folded_conv_names.add(parent_layer.name)
                    print(f"Folded BN '{bn_layer.name}' into Conv '{parent_layer.name}'")
                    
    for layer in unfolded_leafs:
        if layer.__class__.__name__ == 'BatchNormalization':
            continue
        if layer.name in folded_bn_names or layer.name in folded_conv_names:
            continue
        folded_layer = folded_layers.get(layer.name)
        if folded_layer is not None:
            weights = layer.get_weights()
            if weights:
                folded_layer.set_weights(weights)

# =====================================================================
# Target Tracker Ver 5 Class
# =====================================================================

class TargetTrackerVer5:
    def __init__(self, ref_shape=None, search_shape=(256, 256, 1), config_path="model.conf"):
        self.search_shape = search_shape
        self.model = None

        # Load configuration first so ref_shape can be derived from stack params
        resolved_config_path = config_path
        if resolved_config_path and not os.path.isabs(resolved_config_path) and not os.path.exists(resolved_config_path):
            alt_path = os.path.join(os.path.dirname(__file__), resolved_config_path)
            if os.path.exists(alt_path):
                resolved_config_path = alt_path

        if resolved_config_path and os.path.exists(resolved_config_path):
            self.config = load_model_config(resolved_config_path)
        else:
            self.config = {
                "reference_backbone": "mini_mnv2",
                "search_backbone": "mnv2_nano",
                "width_multiplier": 0.5,
                "attention_mechanism": "depthwise_corr",
                "decoder_type": "fpn_add",
                "heatmap_pxl_size": 64,
                "heatmap_output_source": "pre_threshold",
                "heatmap_loss_default": "dbsz_relu",
                "soft_argmax_beta": 15.0,
                "stack_layers": 16,
                "stack_target_size": 64,
                "reference_feature_size": 8,
                "reference_downscale": "auto",
            }

        # Derive ref_shape from config unless explicitly overridden by the caller
        if ref_shape is None:
            sz = self.config.get("stack_target_size", 64)
            ch = self.config.get("stack_layers", 16)
            self.ref_shape = (sz, sz, ch)
        else:
            self.ref_shape = ref_shape

    def _inverted_residual_block(self, inputs, expansion, filters, strides, name_prefix):
        x = inputs
        in_channels = inputs.shape[-1]
        
        # Expand
        if expansion > 1:
            x = layers.Conv2D(expansion * in_channels, (1, 1), padding="same", use_bias=False, name=f"{name_prefix}_expand")(x)
            x = _GroupNormalization(expansion * in_channels, name=f"{name_prefix}_expand_gn")(x)
            x = layers.ReLU(6.0, name=f"{name_prefix}_expand_relu")(x)
            
        # Depthwise
        x = layers.DepthwiseConv2D((3, 3), strides=strides, padding="same", use_bias=False, name=f"{name_prefix}_dw")(x)
        x = _GroupNormalization(expansion * in_channels, name=f"{name_prefix}_dw_gn")(x)
        x = layers.ReLU(6.0, name=f"{name_prefix}_dw_relu")(x)
        
        # Project
        x = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False, name=f"{name_prefix}_project")(x)
        x = _GroupNormalization(filters, name=f"{name_prefix}_project_gn")(x)
        
        if strides == 1 and in_channels == filters:
            x = layers.Add(name=f"{name_prefix}_add")([inputs, x])
            
        return x

    def _conv_norm_relu(self, inputs, filters, kernel_size, strides, name_prefix, relu6=False):
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False, name=f"{name_prefix}_conv")(inputs)
        x = _GroupNormalization(filters, name=f"{name_prefix}_gn")(x)
        if relu6:
            return layers.ReLU(6.0, name=f"{name_prefix}_relu")(x)
        return layers.ReLU(name=f"{name_prefix}_relu")(x)

    def _downscale_reference_features(self, x, bb_type):
        target_size = int(self.config.get("reference_feature_size", 8))
        if target_size <= 0:
            return x

        h = x.shape[1]
        w = x.shape[2]
        if h is None or w is None or int(h) != int(w):
            raise ValueError("Reference feature map must have a known square spatial shape for configured downscale.")

        curr_size = int(h)
        if curr_size < target_size:
            raise ValueError(f"reference_feature_size={target_size} is larger than current reference feature size {curr_size}.")

        mode = self.config.get("reference_downscale", "auto")
        if mode == "auto":
            mode = "depthwise" if bb_type in ("mini_mnv2", "mnv1", "mnv2") else "conv"

        stage = 1
        while curr_size > target_size:
            next_size = curr_size // 2
            if curr_size % 2 != 0 or next_size < target_size:
                raise ValueError(
                    f"reference_feature_size={target_size} is not reachable from {curr_size} "
                    "using repeated stride-2 downscale."
                )

            filters = int(x.shape[-1])
            prefix = f"ref_downscale{stage}"
            if mode == "avg_pool":
                x = layers.AveragePooling2D(pool_size=2, strides=2, name=f"{prefix}_avg_pool")(x)
            elif mode == "max_pool":
                x = layers.MaxPooling2D(pool_size=2, strides=2, name=f"{prefix}_max_pool")(x)
            elif mode == "depthwise":
                x = layers.DepthwiseConv2D((3, 3), strides=2, padding="same", use_bias=False, name=f"{prefix}_dw")(x)
                x = _GroupNormalization(filters, name=f"{prefix}_dw_gn")(x)
                x = layers.ReLU(6.0, name=f"{prefix}_dw_relu")(x)
                x = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False, name=f"{prefix}_pw")(x)
                x = _GroupNormalization(filters, name=f"{prefix}_pw_gn")(x)
                x = layers.ReLU(6.0, name=f"{prefix}_pw_relu")(x)
            elif mode == "conv":
                x = self._conv_norm_relu(x, filters, (3, 3), 2, prefix, relu6=True)
            else:
                raise ValueError(f"Unknown reference_downscale mode: {mode}")

            curr_size = next_size
            stage += 1

        return x

    def _create_search_backbone(self):
        inputs = layers.Input(shape=self.search_shape, name="search_backbone_input")
        
        bb_type = self.config["search_backbone"]
        width_mult = self.config["width_multiplier"]
        
        def scale_filters(f):
            return max(8, int(f * width_mult))
            
        if bb_type == "mnv2_nano":
            # 1. Stride 2 -> (128, 128, 16)
            x1 = layers.Conv2D(scale_filters(16), (3, 3), strides=2, padding="same", use_bias=False, name="sb_init_conv")(inputs)
            x1 = _GroupNormalization(scale_filters(16), name="sb_init_gn")(x1)
            x1 = layers.ReLU(6.0, name="sb_init_relu")(x1)
            
            # 2. Stride 4 -> (64, 64, 16)
            x2 = self._inverted_residual_block(x1, expansion=2, filters=scale_filters(16), strides=2, name_prefix="sb_ir1")
            
            # 3. Stride 8 -> (32, 32, 24)
            x3 = self._inverted_residual_block(x2, expansion=2, filters=scale_filters(24), strides=2, name_prefix="sb_ir2")
            
            # 4. Stride 16 -> (16, 16, 32)
            x4 = self._inverted_residual_block(x3, expansion=3, filters=scale_filters(32), strides=2, name_prefix="sb_ir3")
            
            # Final Expand to attention channels (e.g. 64)
            x = layers.Conv2D(scale_filters(128), (1, 1), padding="same", use_bias=False, name="sb_final_conv")(x4)
            x = _GroupNormalization(scale_filters(128), name="sb_final_gn")(x)
            x = layers.ReLU(6.0, name="sb_final_relu")(x)
            
        elif bb_type == "alex_net":
            # 5 layers of Conv2D + GroupNorm + ReLU
            # Strides: 2, 2, 2, 2, 1
            # Padding: same, so we get 128x128, 64x64, 32x32, 16x16, 16x16
            x1 = layers.Conv2D(scale_filters(16), (11, 11), strides=2, padding="same", use_bias=False, name="sb_init_conv")(inputs)
            x1 = _GroupNormalization(scale_filters(16), name="sb_init_gn")(x1)
            x1 = layers.ReLU(name="sb_init_relu")(x1)
            
            x2 = layers.Conv2D(scale_filters(32), (5, 5), strides=2, padding="same", use_bias=False, name="sb_alex_conv2")(x1)
            x2 = _GroupNormalization(scale_filters(32), name="sb_alex_gn2")(x2)
            x2 = layers.ReLU(name="sb_alex_relu2")(x2)
            
            x3 = layers.Conv2D(scale_filters(64), (3, 3), strides=2, padding="same", use_bias=False, name="sb_alex_conv3")(x2)
            x3 = _GroupNormalization(scale_filters(64), name="sb_alex_gn3")(x3)
            x3 = layers.ReLU(name="sb_alex_relu3")(x3)
            
            x4 = layers.Conv2D(scale_filters(128), (3, 3), strides=2, padding="same", use_bias=False, name="sb_alex_conv4")(x3)
            x4 = _GroupNormalization(scale_filters(128), name="sb_alex_gn4")(x4)
            x4 = layers.ReLU(name="sb_alex_relu4")(x4)
            
            x = layers.Conv2D(scale_filters(128), (3, 3), strides=1, padding="same", use_bias=False, name="sb_alex_conv5")(x4)
            x = _GroupNormalization(scale_filters(128), name="sb_alex_gn5")(x)
            x = layers.ReLU(name="sb_alex_relu5")(x)

        elif bb_type == "alex_dense":
            x1 = self._conv_norm_relu(inputs, scale_filters(16), (7, 7), 2, "sb_dense1")
            x2 = self._conv_norm_relu(x1, scale_filters(32), (5, 5), 2, "sb_dense2")
            x3 = self._conv_norm_relu(x2, scale_filters(64), (3, 3), 2, "sb_dense3")
            x4 = self._conv_norm_relu(x3, scale_filters(128), (3, 3), 2, "sb_dense4")
            x = self._conv_norm_relu(x4, scale_filters(128), (3, 3), 1, "sb_dense5")

        elif bb_type == "mnv1":
            # MobileNetV1 style with skip connections
            # Stride 2 -> (128, 128, 16)
            x1 = layers.Conv2D(scale_filters(16), (3, 3), strides=2, padding="same", use_bias=False, name="sb_init_conv")(inputs)
            x1 = _GroupNormalization(scale_filters(16), name="sb_init_gn")(x1)
            x1 = layers.ReLU(6.0, name="sb_init_relu")(x1)
            
            # Stride 4 -> (64, 64, 24)
            x = layers.DepthwiseConv2D((3, 3), strides=2, padding="same", use_bias=False, name="sb_dw1")(x1)
            x = _GroupNormalization(scale_filters(16), name="sb_dw1_gn")(x)
            x = layers.ReLU(6.0, name="sb_dw1_relu")(x)
            x2 = layers.Conv2D(scale_filters(24), (1, 1), padding="same", use_bias=False, name="sb_pw1")(x)
            x2 = _GroupNormalization(scale_filters(24), name="sb_pw1_gn")(x2)
            x2 = layers.ReLU(6.0, name="sb_pw1_relu")(x2)
            
            # Stride 8 -> (32, 32, 32)
            x = layers.DepthwiseConv2D((3, 3), strides=2, padding="same", use_bias=False, name="sb_dw2")(x2)
            x = _GroupNormalization(scale_filters(24), name="sb_dw2_gn")(x)
            x = layers.ReLU(6.0, name="sb_dw2_relu")(x)
            x3 = layers.Conv2D(scale_filters(32), (1, 1), padding="same", use_bias=False, name="sb_pw2")(x)
            x3 = _GroupNormalization(scale_filters(32), name="sb_pw2_gn")(x3)
            x3 = layers.ReLU(6.0, name="sb_pw2_relu")(x3)
            
            # Stride 16 -> (16, 16, 64)
            x = layers.DepthwiseConv2D((3, 3), strides=2, padding="same", use_bias=False, name="sb_dw3")(x3)
            x = _GroupNormalization(scale_filters(32), name="sb_dw3_gn")(x)
            x = layers.ReLU(6.0, name="sb_dw3_relu")(x)
            x4 = layers.Conv2D(scale_filters(64), (1, 1), padding="same", use_bias=False, name="sb_pw3")(x)
            x4 = _GroupNormalization(scale_filters(64), name="sb_pw3_gn")(x4)
            x4 = layers.ReLU(6.0, name="sb_pw3_relu")(x4)
            
            # Final Expand
            x = layers.Conv2D(scale_filters(128), (1, 1), padding="same", use_bias=False, name="sb_final_conv")(x4)
            x = _GroupNormalization(scale_filters(128), name="sb_final_gn")(x)
            x = layers.ReLU(6.0, name="sb_final_relu")(x)
            
        elif bb_type == "mnv2":
            # Full MobileNetV2
            x1 = layers.Conv2D(scale_filters(16), (3, 3), strides=2, padding="same", use_bias=False, name="sb_init_conv")(inputs)
            x1 = _GroupNormalization(scale_filters(16), name="sb_init_gn")(x1)
            x1 = layers.ReLU(6.0, name="sb_init_relu")(x1)
            
            x2 = self._inverted_residual_block(x1, expansion=2, filters=scale_filters(24), strides=2, name_prefix="sb_ir1")
            x3 = self._inverted_residual_block(x2, expansion=3, filters=scale_filters(32), strides=2, name_prefix="sb_ir2")
            x4 = self._inverted_residual_block(x3, expansion=4, filters=scale_filters(64), strides=2, name_prefix="sb_ir3")
            
            x = layers.Conv2D(scale_filters(128), (1, 1), padding="same", use_bias=False, name="sb_final_conv")(x4)
            x = _GroupNormalization(scale_filters(128), name="sb_final_gn")(x)
            x = layers.ReLU(6.0, name="sb_final_relu")(x)
            
        elif bb_type == "yolo5":
            # YOLOv5-style CSP blocks
            # Stride 2 -> (128, 128, 16)
            x1 = layers.Conv2D(scale_filters(16), (3, 3), strides=2, padding="same", use_bias=False, name="sb_init_conv")(inputs)
            x1 = _GroupNormalization(scale_filters(16), name="sb_init_gn")(x1)
            x1 = layers.ReLU(6.0, name="sb_init_relu")(x1)
            
            # Stride 4 -> (64, 64, 24)
            x2 = layers.Conv2D(scale_filters(24), (3, 3), strides=2, padding="same", use_bias=False, name="sb_ir1_conv")(x1)
            x2 = _GroupNormalization(scale_filters(24), name="sb_ir1_gn")(x2)
            x2 = layers.ReLU(6.0, name="sb_ir1_relu")(x2)
            
            # Stride 8 -> (32, 32, 32)
            x3 = layers.Conv2D(scale_filters(32), (3, 3), strides=2, padding="same", use_bias=False, name="sb_ir2_conv")(x2)
            x3 = _GroupNormalization(scale_filters(32), name="sb_ir2_gn")(x3)
            x3 = layers.ReLU(6.0, name="sb_ir2_relu")(x3)
            
            # Stride 16 -> (16, 16, 64)
            x4 = layers.Conv2D(scale_filters(64), (3, 3), strides=2, padding="same", use_bias=False, name="sb_ir3_conv")(x3)
            x4 = _GroupNormalization(scale_filters(64), name="sb_ir3_gn")(x4)
            x4 = layers.ReLU(6.0, name="sb_ir3_relu")(x4)
            
            # Final Expand
            x = layers.Conv2D(scale_filters(128), (1, 1), padding="same", use_bias=False, name="sb_final_conv")(x4)
            x = _GroupNormalization(scale_filters(128), name="sb_final_gn")(x)
            x = layers.ReLU(6.0, name="sb_final_relu")(x)
            
        elif bb_type == "unet":
            # UNet-style encoder: 2 conv blocks per resolution, strided-conv for downsampling (NPU-friendly, no MaxPool)
            # Stage 1: stride 2 -> (128, 128, C16)
            x1 = layers.Conv2D(scale_filters(16), (3, 3), strides=2, padding="same", use_bias=False, name="sb_ue1_conv1")(inputs)
            x1 = _GroupNormalization(scale_filters(16), name="sb_ue1_gn1")(x1)
            x1 = layers.ReLU(6.0, name="sb_ue1_relu1")(x1)
            x1 = layers.Conv2D(scale_filters(16), (7, 7), strides=1, padding="same", use_bias=False, name="sb_ue1_conv2")(x1)
            x1 = _GroupNormalization(scale_filters(16), name="sb_ue1_gn2")(x1)
            x1 = layers.ReLU(6.0, name="sb_ue1_relu2")(x1)
            
            # Stage 2: stride 4 -> (64, 64, C24)
            x2 = layers.Conv2D(scale_filters(24), (3, 3), strides=2, padding="same", use_bias=False, name="sb_ue2_conv1")(x1)
            x2 = _GroupNormalization(scale_filters(24), name="sb_ue2_gn1")(x2)
            x2 = layers.ReLU(6.0, name="sb_ue2_relu1")(x2)
            x2 = layers.Conv2D(scale_filters(24), (5, 5), strides=1, padding="same", use_bias=False, name="sb_ue2_conv2")(x2)
            x2 = _GroupNormalization(scale_filters(24), name="sb_ue2_gn2")(x2)
            x2 = layers.ReLU(6.0, name="sb_ue2_relu2")(x2)
            
            # Stage 3: stride 8 -> (32, 32, C32)
            x3 = layers.Conv2D(scale_filters(32), (3, 3), strides=2, padding="same", use_bias=False, name="sb_ue3_conv1")(x2)
            x3 = _GroupNormalization(scale_filters(32), name="sb_ue3_gn1")(x3)
            x3 = layers.ReLU(6.0, name="sb_ue3_relu1")(x3)
            x3 = layers.Conv2D(scale_filters(32), (3, 3), strides=1, padding="same", use_bias=False, name="sb_ue3_conv2")(x3)
            x3 = _GroupNormalization(scale_filters(32), name="sb_ue3_gn2")(x3)
            x3 = layers.ReLU(6.0, name="sb_ue3_relu2")(x3)
            
            # Bottleneck: stride 16 -> (16, 16, C128)
            x = layers.Conv2D(scale_filters(64), (3, 3), strides=2, padding="same", use_bias=False, name="sb_ue4_conv1")(x3)
            x = _GroupNormalization(scale_filters(64), name="sb_ue4_gn1")(x)
            x = layers.ReLU(6.0, name="sb_ue4_relu1")(x)
            x = layers.Conv2D(scale_filters(128), (1, 1), padding="same", use_bias=False, name="sb_ue4_conv2")(x)
            x = _GroupNormalization(scale_filters(128), name="sb_ue4_gn2")(x)
            x = layers.ReLU(6.0, name="sb_ue4_relu2")(x)
            
        else: # custom_legacy
            # Init conv (strides=2) -> (128, 128, 16)
            x1 = layers.Conv2D(16, (3, 3), strides=2, padding="same", use_bias=False, name="sb_init_conv")(inputs)
            x1 = _GroupNormalization(16, name="sb_init_gn")(x1)
            x1 = layers.ReLU(6.0, name="sb_init_relu")(x1)
            
            # IR Block 1 (strides=2) -> (64, 64, 24)
            x2 = self._inverted_residual_block(x1, expansion=2, filters=24, strides=2, name_prefix="sb_ir1")
            
            # IR Block 2 (strides=2) -> (32, 32, 32)
            x3 = self._inverted_residual_block(x2, expansion=3, filters=32, strides=2, name_prefix="sb_ir2")
            
            # IR Block 3 (strides=2) -> (16, 16, 64)
            x4 = self._inverted_residual_block(x3, expansion=4, filters=64, strides=2, name_prefix="sb_ir3")
            
            # Final Expand -> (16, 16, 128)
            x = layers.Conv2D(128, (1, 1), padding="same", use_bias=False, name="sb_final_conv")(x4)
            x = _GroupNormalization(128, name="sb_final_gn")(x)
            x = layers.ReLU(6.0, name="sb_final_relu")(x)
            
        return models.Model(inputs, [x1, x2, x3, x], name="search_feature_extractor")

    def _create_reference_encoder(self):
        inputs = layers.Input(shape=self.ref_shape, name="ref_encoder_input")
        
        # 1. Reshape dynamically based on the input structure
        if len(self.ref_shape) == 3:
            # Native 4D format: (64, 64, 16)
            x = inputs
        elif self.ref_shape[0] == 1:
            # New format: (1, 64, 64, 16) -> (64, 64, 16)
            x = layers.Reshape((self.ref_shape[1], self.ref_shape[2], self.ref_shape[3]), name="ref_reshape")(inputs)
        else:
            # Legacy format: (16, 32, 32, 1) -> (32, 32, 16)
            x = layers.Permute((2, 3, 1, 4), name="ref_permute")(inputs)
            x = layers.Reshape((self.ref_shape[1], self.ref_shape[2], self.ref_shape[0] * self.ref_shape[3]), name="ref_reshape")(x)
        
        bb_type = self.config["reference_backbone"]
        width_mult = self.config["width_multiplier"]
        
        def scale_filters(f):
            return max(8, int(f * width_mult))
            
        if bb_type == "alex_net":
            # 5 layers of Conv2D + GroupNorm + ReLU
            # Input shape: 64x64x16
            # Strides: 2, 2, 2, 1, 1 (Total stride 8)
            # Padding: same, so we get 32x32, 16x16, 8x8, 8x8, 8x8
            x = layers.Conv2D(scale_filters(16), (5, 5), strides=2, padding="same", use_bias=False, name="ref_alex_conv1")(x)
            x = _GroupNormalization(scale_filters(16), name="ref_alex_gn1")(x)
            x = layers.ReLU(name="ref_alex_relu1")(x)
            
            x = layers.Conv2D(scale_filters(32), (3, 3), strides=2, padding="same", use_bias=False, name="ref_alex_conv2")(x)
            x = _GroupNormalization(scale_filters(32), name="ref_alex_gn2")(x)
            x = layers.ReLU(name="ref_alex_relu2")(x)
            
            x = layers.Conv2D(scale_filters(64), (3, 3), strides=2, padding="same", use_bias=False, name="ref_alex_conv3")(x)
            x = _GroupNormalization(scale_filters(64), name="ref_alex_gn3")(x)
            x = layers.ReLU(name="ref_alex_relu3")(x)
            
            x = layers.Conv2D(scale_filters(128), (3, 3), strides=1, padding="same", use_bias=False, name="ref_alex_conv4")(x)
            x = _GroupNormalization(scale_filters(128), name="ref_alex_gn4")(x)
            x = layers.ReLU(name="ref_alex_relu4")(x)
            
            out_channels = scale_filters(128)
            x = layers.Conv2D(out_channels, (3, 3), strides=1, padding="same", use_bias=False, name="ref_alex_conv5")(x)
            x = _GroupNormalization(out_channels, name="ref_alex_gn5")(x)
            x = layers.ReLU(name="ref_alex_relu5")(x)
            
        elif bb_type == "alex_dense":
            x = self._conv_norm_relu(x, scale_filters(16), (7, 7), 2, "ref_dense1")
            x = self._conv_norm_relu(x, scale_filters(32), (5, 5), 2, "ref_dense2")
            x = self._conv_norm_relu(x, scale_filters(64), (3, 3), 2, "ref_dense3")
            out_channels = scale_filters(128)
            x = self._conv_norm_relu(x, out_channels, (3, 3), 1, "ref_dense4")
            x = self._conv_norm_relu(x, out_channels, (3, 3), 1, "ref_dense5")
            
        elif bb_type == "mini_mnv2":
            # For 64x64 input, stride=2 on init conv to get 8x8 output (strides 2 * 2 * 2 = 8)
            x = layers.Conv2D(scale_filters(16), (3, 3), strides=2, padding="same", use_bias=False, name="ref_init_conv")(x)
            x = _GroupNormalization(scale_filters(16), name="ref_init_gn")(x)
            x = layers.ReLU(6.0, name="ref_init_relu")(x)
            
            x = self._inverted_residual_block(x, expansion=2, filters=scale_filters(24), strides=2, name_prefix="ref_ir1")
            x = self._inverted_residual_block(x, expansion=2, filters=scale_filters(32), strides=2, name_prefix="ref_ir2")
            
            out_channels = scale_filters(128)
            x = layers.Conv2D(out_channels, (1, 1), padding="same", use_bias=False, name="ref_final_conv")(x)
            x = _GroupNormalization(out_channels, name="ref_final_gn")(x)
            x = layers.ReLU(6.0, name="ref_final_relu")(x)
            
        elif bb_type == "mnv1":
            # Strides: 2 * 2 * 2 = 8
            x = layers.Conv2D(scale_filters(16), (3, 3), strides=2, padding="same", use_bias=False, name="ref_init_conv")(x)
            x = _GroupNormalization(scale_filters(16), name="ref_init_gn")(x)
            x = layers.ReLU(6.0, name="ref_init_relu")(x)
            
            x = layers.DepthwiseConv2D((3, 3), strides=2, padding="same", use_bias=False, name="ref_dw1")(x)
            x = _GroupNormalization(scale_filters(16), name="ref_dw1_gn")(x)
            x = layers.ReLU(6.0, name="ref_dw1_relu")(x)
            x = layers.Conv2D(scale_filters(32), (1, 1), padding="same", use_bias=False, name="ref_pw1")(x)
            x = _GroupNormalization(scale_filters(32), name="ref_pw1_gn")(x)
            x = layers.ReLU(6.0, name="ref_pw1_relu")(x)
            
            # Additional block to get 8x downsampling total
            x = layers.DepthwiseConv2D((3, 3), strides=2, padding="same", use_bias=False, name="ref_dw2")(x)
            x = _GroupNormalization(scale_filters(32), name="ref_dw2_gn")(x)
            x = layers.ReLU(6.0, name="ref_dw2_relu")(x)
            x = layers.Conv2D(scale_filters(64), (1, 1), padding="same", use_bias=False, name="ref_pw2")(x)
            x = _GroupNormalization(scale_filters(64), name="ref_pw2_gn")(x)
            x = layers.ReLU(6.0, name="ref_pw2_relu")(x)
            
            out_channels = scale_filters(128)
            x = layers.Conv2D(out_channels, (1, 1), padding="same", use_bias=False, name="ref_final_conv")(x)
            x = _GroupNormalization(out_channels, name="ref_final_gn")(x)
            x = layers.ReLU(6.0, name="ref_final_relu")(x)
            
        elif bb_type == "mnv2":
            # Strides: 2 * 2 * 2 = 8
            x = layers.Conv2D(scale_filters(16), (3, 3), strides=2, padding="same", use_bias=False, name="ref_init_conv")(x)
            x = _GroupNormalization(scale_filters(16), name="ref_init_gn")(x)
            x = layers.ReLU(6.0, name="ref_init_relu")(x)
            
            x = self._inverted_residual_block(x, expansion=2, filters=scale_filters(32), strides=2, name_prefix="ref_ir1")
            
            # Additional block to get 8x downsampling total
            x = self._inverted_residual_block(x, expansion=2, filters=scale_filters(64), strides=2, name_prefix="ref_ir2")
            
            out_channels = scale_filters(128)
            x = layers.Conv2D(out_channels, (1, 1), padding="same", use_bias=False, name="ref_final_conv")(x)
            x = _GroupNormalization(out_channels, name="ref_final_gn")(x)
            x = layers.ReLU(6.0, name="ref_final_relu")(x)
            
        elif bb_type == "yolo5":
            # Strides: 2 * 2 * 2 = 8
            x = layers.Conv2D(scale_filters(16), (3, 3), strides=2, padding="same", use_bias=False, name="ref_init_conv")(x)
            x = _GroupNormalization(scale_filters(16), name="ref_init_gn")(x)
            x = layers.ReLU(6.0, name="ref_init_relu")(x)
            
            # Additional downsampling step to get 8x downsampling total
            x = layers.Conv2D(scale_filters(32), (3, 3), strides=2, padding="same", use_bias=False, name="ref_down2")(x)
            x = _GroupNormalization(scale_filters(32), name="ref_down2_gn")(x)
            x = layers.ReLU(6.0, name="ref_down2_relu")(x)
            
            c_half = scale_filters(32)
            x1 = layers.Conv2D(c_half, (1, 1), padding="same", use_bias=False, name="ref_csp1")(x)
            x2 = layers.Conv2D(c_half, (1, 1), padding="same", use_bias=False, name="ref_csp2")(x)
            x1 = layers.Conv2D(c_half, (3, 3), strides=2, padding="same", use_bias=False, name="ref_csp_conv")(x1)
            x1 = _GroupNormalization(c_half, name="ref_csp_gn")(x1)
            x1 = layers.ReLU(6.0, name="ref_csp_relu")(x1)
            x2 = layers.AveragePooling2D(pool_size=2, name="ref_csp_pool")(x2)
            
            x = layers.Concatenate(axis=-1, name="ref_csp_cat")([x1, x2])
            out_channels = scale_filters(128)
            x = layers.Conv2D(out_channels, (1, 1), padding="same", use_bias=False, name="ref_final_conv")(x)
            x = _GroupNormalization(out_channels, name="ref_final_gn")(x)
            x = layers.ReLU(6.0, name="ref_final_relu")(x)
            
        elif bb_type == "unet":
            # UNet-style encoder: 2 conv blocks per resolution, strided-conv for downsampling (NPU-friendly, no MaxPool)
            # Stage 1: stride 2 -> (32, 32, C16)
            x = layers.Conv2D(scale_filters(16), (3, 3), strides=2, padding="same", use_bias=False, name="ref_ue1_conv1")(x)
            x = _GroupNormalization(scale_filters(16), name="ref_ue1_gn1")(x)
            x = layers.ReLU(6.0, name="ref_ue1_relu1")(x)
            x = layers.Conv2D(scale_filters(16), (5, 5), strides=1, padding="same", use_bias=False, name="ref_ue1_conv2")(x)
            x = _GroupNormalization(scale_filters(16), name="ref_ue1_gn2")(x)
            x = layers.ReLU(6.0, name="ref_ue1_relu2")(x)
            
            # Stage 2: stride 4 -> (16, 16, C32)
            x = layers.Conv2D(scale_filters(32), (3, 3), strides=2, padding="same", use_bias=False, name="ref_ue2_conv1")(x)
            x = _GroupNormalization(scale_filters(32), name="ref_ue2_gn1")(x)
            x = layers.ReLU(6.0, name="ref_ue2_relu1")(x)
            x = layers.Conv2D(scale_filters(32), (3, 3), strides=1, padding="same", use_bias=False, name="ref_ue2_conv2")(x)
            x = _GroupNormalization(scale_filters(32), name="ref_ue2_gn2")(x)
            x = layers.ReLU(6.0, name="ref_ue2_relu2")(x)
            
            # Bottleneck: stride 8 -> (8, 8, C128)
            x = layers.Conv2D(scale_filters(64), (3, 3), strides=2, padding="same", use_bias=False, name="ref_ue3_conv1")(x)
            x = _GroupNormalization(scale_filters(64), name="ref_ue3_gn1")(x)
            x = layers.ReLU(6.0, name="ref_ue3_relu1")(x)
            x = layers.Conv2D(scale_filters(128), (1, 1), padding="same", use_bias=False, name="ref_ue3_conv2")(x)
            x = _GroupNormalization(scale_filters(128), name="ref_ue3_gn2")(x)
            x = layers.ReLU(6.0, name="ref_ue3_relu2")(x)
            
        else: # custom_legacy
            # Strides: 2 * 2 * 2 = 8
            x = layers.Conv2D(32, (3, 3), strides=2, padding="same", use_bias=False, name="ref_init_conv")(x)
            x = _GroupNormalization(32, name="ref_init_gn")(x)
            x = layers.ReLU(6.0, name="ref_init_relu")(x)
            x = self._inverted_residual_block(x, expansion=2, filters=64, strides=2, name_prefix="ref_ir1")
            x = self._inverted_residual_block(x, expansion=2, filters=96, strides=2, name_prefix="ref_ir2")
            x = layers.Conv2D(128, (1, 1), padding="same", use_bias=False, name="ref_final_conv")(x)
            x = _GroupNormalization(128, name="ref_final_gn")(x)
            x = layers.ReLU(6.0, name="ref_final_relu")(x)
            
        x = self._downscale_reference_features(x, bb_type)
        return models.Model(inputs, x, name="reference_target_encoder")

    def create_model(self):
        global _NORMALIZATION_TYPE, _OVERRIDE_USE_BIAS
        _NORMALIZATION_TYPE = self.config.get("normalization_type", "group_norm")
        _OVERRIDE_USE_BIAS = True
        try:
            return self._create_model_impl()
        finally:
            _OVERRIDE_USE_BIAS = False

    def _create_model_impl(self):
        ref_input = layers.Input(shape=self.ref_shape, name="reference_stack")
        search_input = layers.Input(shape=self.search_shape, name="search_frame")
        
        # 1. Encoders
        ref_encoder = self._create_reference_encoder()
        search_encoder = self._create_search_backbone()
        
        ref_features = ref_encoder(ref_input)
        
        # Multi-output Search Encoder to get intermediate features for skip connections
        sb_init, sb_ir1, sb_ir2, search_features = search_encoder(search_input)
        
        bb_type = self.config["search_backbone"]
        width_mult = self.config["width_multiplier"]
        
        def scale_filters(f):
            return max(8, int(f * width_mult))
            
        attn_mech = self.config["attention_mechanism"]
        
        # 2. Attention Fusion
        if attn_mech == "depthwise_corr":
            fused_features = DepthwiseCorrelationFusion(name="depthwise_correlation_fusion")([search_features, ref_features])
            c_v = scale_filters(128) if bb_type != "custom_legacy" else 128
            
        elif attn_mech == "conv2d_corr":
            c_v = scale_filters(128) if bb_type != "custom_legacy" else 128
            fused_features = Conv2DCorrelationFusion(channels=c_v, name="conv2d_correlation_fusion")([search_features, ref_features])
            
        elif attn_mech == "linear_cross":
            c_qk = scale_filters(64) if bb_type != "custom_legacy" else 64
            c_v = scale_filters(128) if bb_type != "custom_legacy" else 128
            
            q_proj = layers.Conv2D(c_qk, (1, 1), use_bias=False, name="q_proj")(search_features)
            k_proj = layers.Conv2D(c_qk, (1, 1), use_bias=False, name="k_proj")(ref_features)
            v_proj = layers.Conv2D(c_v, (1, 1), use_bias=False, name="v_proj")(ref_features)
            
            q_flat = layers.Reshape((256, c_qk), name="q_flat")(q_proj)
            k_flat = layers.Reshape((64, c_qk), name="k_flat")(k_proj)
            v_flat = layers.Reshape((64, c_v), name="v_flat")(v_proj)
            
            q_soft = layers.Softmax(axis=-1, name="q_softmax")(q_flat)
            k_soft = layers.Softmax(axis=1, name="k_softmax")(k_flat)
            
            k_trans = layers.Permute((2, 1), name="k_trans")(k_soft)
            kv = layers.Dot(axes=(2, 1), name="kv_dot")([k_trans, v_flat])
            fused_flat = layers.Dot(axes=(2, 1), name="linear_attention_dot")([q_soft, kv])
            fused_features = layers.Reshape((16, 16, c_v), name="fused_features_reshape")(fused_flat)
            
        elif attn_mech == "multi_head_cross":
            c_v = scale_filters(128) if bb_type != "custom_legacy" else 128
            q_flat = layers.Reshape((256, c_v), name="q_flat")(search_features)
            kv_flat = layers.Reshape((64, c_v), name="kv_flat")(ref_features)
            
            mha = layers.MultiHeadAttention(num_heads=4, key_dim=c_v // 4, name="mha_fusion")
            fused_flat = mha(query=q_flat, value=kv_flat, key=kv_flat)
            fused_features = layers.Reshape((16, 16, c_v), name="fused_features_reshape")(fused_flat)
            
        else: # dot_cross
            c_qk = scale_filters(64) if bb_type != "custom_legacy" else 64
            c_v = scale_filters(128) if bb_type != "custom_legacy" else 128
            
            q_proj = layers.Conv2D(c_qk, (1, 1), use_bias=False, name="q_proj")(search_features)
            k_proj = layers.Conv2D(c_qk, (1, 1), use_bias=False, name="k_proj")(ref_features)
            v_proj = layers.Conv2D(c_v, (1, 1), use_bias=False, name="v_proj")(ref_features)
            
            q_flat = layers.Reshape((256, c_qk), name="q_flat")(q_proj)
            k_flat = layers.Reshape((64, c_qk), name="k_flat")(k_proj)
            v_flat = layers.Reshape((64, c_v), name="v_flat")(v_proj)
            
            attn_weights = layers.Dot(axes=(2, 2), name="attention_dot")([q_flat, k_flat])
            attn_weights = layers.Rescaling(scale=1.0 / np.sqrt(c_qk), name="attention_scale")(attn_weights)
            attn_weights = layers.Softmax(axis=-1, name="attention_softmax")(attn_weights)
            
            fused_flat = layers.Dot(axes=(2, 1), name="attention_value_dot")([attn_weights, v_flat])
            fused_features = layers.Reshape((16, 16, c_v), name="fused_features_reshape")(fused_flat)
            
        # 3. Decoder for Output 1: Heatmap
        dec_type = self.config["decoder_type"]
        heatmap_pxl_size = self.config.get("heatmap_pxl_size", 64)
        
        curr_size = 16
        x = fused_features
        skips = [sb_ir2, sb_ir1, sb_init]
        stage = 1
        
        if dec_type == "pixel_shuffle":
            def pixel_shuffle_block(inputs, out_filters, name_prefix):
                ps_c = out_filters * 4
                x_ps = layers.Conv2D(ps_c, (3, 3), padding="same", use_bias=False, name=f"{name_prefix}_ps_conv")(inputs)
                x_ps = _GroupNormalization(ps_c, name=f"{name_prefix}_ps_gn")(x_ps)
                x_ps = layers.ReLU(6.0, name=f"{name_prefix}_ps_relu")(x_ps)
                x_ps = DepthToSpace(block_size=2, name=f"{name_prefix}_ps_shuffle")(x_ps)
                return x_ps
        
        while curr_size < heatmap_pxl_size:
            curr_size *= 2
            
            if dec_type == "fpn_add":
                x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"decoder_up{stage}")(x)
                if stage <= len(skips):
                    skip = skips[stage - 1]
                    skip_proj = layers.Conv2D(x.shape[-1], (1, 1), padding="same", use_bias=False, name=f"decoder_skip{stage}_proj")(skip)
                    x = layers.Add(name=f"decoder_skip{stage}")([x, skip_proj])
                x = self._inverted_residual_block(x, expansion=2, filters=scale_filters(64 // (2**(stage-1))) if bb_type != "custom_legacy" else (64 // (2**(stage-1))), strides=1, name_prefix=f"decoder_ir{stage}")
                
            elif dec_type == "pixel_shuffle":
                filters = scale_filters(64 // (2**(stage-1))) if bb_type != "custom_legacy" else (64 // (2**(stage-1)))
                x = pixel_shuffle_block(x, filters, f"decoder{stage}")
                if stage <= len(skips):
                    skip = skips[stage - 1]
                    skip_proj = layers.Conv2D(x.shape[-1], (1, 1), padding="same", use_bias=False, name=f"decoder_skip{stage}_proj")(skip)
                    x = layers.Add(name=f"decoder_skip{stage}")([x, skip_proj])
                    
            elif dec_type == "light_naive":
                filters = scale_filters(64 // (2**(stage-1))) if bb_type != "custom_legacy" else (64 // (2**(stage-1)))
                x = layers.Conv2DTranspose(filters, (3, 3), strides=2, padding="same", use_bias=False, name=f"decoder_up{stage}")(x)
                x = _GroupNormalization(filters, name=f"decoder_up{stage}_gn")(x)
                x = layers.ReLU(6.0, name=f"decoder_up{stage}_relu")(x)
                
            else: # unet
                x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"decoder_up{stage}")(x)
                if stage <= len(skips):
                    skip = skips[stage - 1]
                    x = layers.Concatenate(axis=-1, name=f"decoder_skip{stage}")([x, skip])
                x = self._inverted_residual_block(x, expansion=2, filters=scale_filters(64 // (2**(stage-1))) if bb_type != "custom_legacy" else (64 // (2**(stage-1))), strides=1, name_prefix=f"decoder_ir{stage}")
                
            stage += 1
            
        # Final prediction heatmap
        output_heatmap_raw = layers.Conv2D(1, (3, 3), padding="same", activation="relu", name="predicted_heatmap_raw")(x)
        output_heatmap_norm = layers.BatchNormalization(name="predicted_heatmap_norm")(output_heatmap_raw)
        output_heatmap_pre_threshold = layers.Rescaling(scale=2.0, offset=-1.0, name="heatmap_threshold_rescale")(output_heatmap_norm)
        thresholded = layers.ReLU(name="heatmap_threshold_relu")(output_heatmap_pre_threshold)
        output_heatmap_processed = layers.DepthwiseConv2D(
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="same",
            depth_multiplier=1,
            use_bias=False,
            depthwise_initializer=tf.keras.initializers.Constant(1.0 / 25.0),
            trainable=False,
            name="predicted_heatmap_processed"
        )(thresholded)

        heatmap_output_source = self.config.get("heatmap_output_source", "pre_threshold")
        if heatmap_output_source == "raw":
            heatmap_for_loss = output_heatmap_raw
        elif heatmap_output_source == "norm":
            heatmap_for_loss = output_heatmap_norm
        elif heatmap_output_source == "pre_threshold":
            heatmap_for_loss = output_heatmap_pre_threshold
        elif heatmap_output_source == "processed":
            heatmap_for_loss = output_heatmap_processed
        else:
            raise ValueError(f"Unknown heatmap_output_source: {heatmap_output_source}")

        output_heatmap = layers.Activation("linear", name="predicted_heatmap")(heatmap_for_loss)
        
        # 4. Heatmap-Guided Classification Branch
        ds_factor = heatmap_pxl_size // 16
        if ds_factor >= 2:
            hm_feat = layers.Conv2D(8, (3, 3), strides=2, padding="same", activation="relu", name="quality_hm_conv")(output_heatmap_processed)
            pool_size = ds_factor // 2
            if pool_size > 1:
                hm_pool = layers.AveragePooling2D(pool_size=pool_size, name="quality_hm_pool")(hm_feat)
            else:
                hm_pool = hm_feat
        else:
            hm_feat = layers.Conv2D(8, (3, 3), strides=1, padding="same", activation="relu", name="quality_hm_conv")(output_heatmap_processed)
            hm_pool = hm_feat
            
        q_fused = layers.Concatenate(axis=-1, name="quality_fusion")([fused_features, hm_pool])
        
        q = layers.Conv2D(64, (3, 3), strides=2, padding="same", use_bias=False, name="quality_conv1")(q_fused)
        q = _GroupNormalization(64, name="quality_gn1")(q)
        q = layers.ReLU(6.0, name="quality_relu1")(q)
        
        q = layers.Conv2D(32, (3, 3), strides=2, padding="same", use_bias=False, name="quality_conv2")(q)
        q = _GroupNormalization(32, name="quality_gn2")(q)
        q = layers.ReLU(6.0, name="quality_relu2")(q)
        
        q = layers.GlobalAveragePooling2D(name="quality_gap")(q)
        q = layers.Dense(16, activation="relu", name="quality_fc1")(q)
        output_quality = layers.Dense(1, activation="sigmoid", name="predicted_quality")(q)
        
        self.model = models.Model(
            inputs=[ref_input, search_input],
            outputs=[output_heatmap, output_quality],
            name="TargetTrackerVer5"
        )
        
        print("\\n" + "="*50)
        print("Model Architectural Configuration:")
        for k, v in self.config.items():
            print(f"  {k}: {v}")
        print("="*50)
        self.model.summary()
        print(f"Total Model Parameters: {self.model.count_params():,}")
        print("="*50 + "\\n")
        
        return self.model

    def fold_and_save(self, folded_output_path):
        if self.model is None:
            raise ValueError("Model is not built yet.")
            
        import copy
        folded_tracker = self.__class__(ref_shape=self.ref_shape, search_shape=self.search_shape)
        folded_tracker.config = copy.deepcopy(self.config)
        folded_tracker.config["normalization_type"] = "folded"
        
        folded_model = folded_tracker.create_model()
        fold_model_weights(self.model, folded_model)
        
        if os.path.dirname(folded_output_path):
            os.makedirs(os.path.dirname(folded_output_path), exist_ok=True)
        folded_model.save(folded_output_path)
        print(f"Saved folded model to {folded_output_path}")

    @staticmethod
    def _shape_to_list(shape):
        if hasattr(shape, "as_list"):
            shape = shape.as_list()
        return [None if dim is None else int(dim) for dim in shape]

    def get_runtime_config(self, pipeline_config_path="dataset_generator/pipeline_config.json"):
        if self.model is None:
            raise ValueError("Model is not built yet.")

        compiler_cfg = {}
        if pipeline_config_path and os.path.exists(pipeline_config_path):
            with open(pipeline_config_path, "r", encoding="utf-8") as f:
                compiler_cfg = json.load(f).get("compiler", {})

        ref_encoder = self.model.get_layer("reference_target_encoder")
        processed_hm_layer = self.model.get_layer("predicted_heatmap_processed")

        return {
            "runtime_config_version": 1,
            "model_name": self.model.name,
            "model_config": dict(self.config),
            "model_input": {
                "reference_stack_shape": list(self.ref_shape),
                "search_frame_shape": list(self.search_shape),
                "heatmap_output_shape": self._shape_to_list(self.model.outputs[0].shape[1:]),
                "processed_heatmap_shape": self._shape_to_list(processed_hm_layer.output.shape[1:]),
                "reference_feature_shape": self._shape_to_list(ref_encoder.output_shape[1:]),
            },
            "preprocessing": {
                "search_crop_mode": "center_square",
                "search_crop_ratio": 1.0,
                "search_resize": "bilinear",
                "reference_resize": "bilinear",
                "input_range": "0_1_training_uint8_rknn",
            },
            "reference_stack": {
                "stack_layers": self.config.get("stack_layers"),
                "stack_target_size": self.config.get("stack_target_size"),
                "crop_min_size": compiler_cfg.get("crop_min_size"),
                "crop_max_size": compiler_cfg.get("crop_max_size"),
            },
            "postprocessing": {
                "model_heatmap_output": self.config.get("heatmap_output_source", "pre_threshold"),
                "processed_heatmap_tensor": "predicted_heatmap_processed",
                "decode": "argmax_or_local_centroid",
                "coordinate_scale": 256,
            },
        }

    def save_runtime_config(self, output_path, pipeline_config_path="dataset_generator/pipeline_config.json"):
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.get_runtime_config(pipeline_config_path), f, indent=2)
            f.write("\n")
        print(f"Saved runtime config to {output_path}")

    def log(self, message, log_file=None):
        import tqdm
        tqdm.tqdm.write(message)
        if log_file:
            parent_dir = os.path.dirname(log_file)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(message + "\n")

    def evaluate(self, dataset, loss_fn_quality, train_mode="joint", steps=None):
        val_loss_avg = tf.keras.metrics.Mean()
        val_hm_loss_avg = tf.keras.metrics.Mean()
        val_q_loss_avg = tf.keras.metrics.Mean()
        import tqdm
        
        progress_bar = tqdm.tqdm(dataset, desc="Evaluating", total=steps, leave=False)
        for inputs, targets in progress_bar:
            gt_coords = targets["predicted_coords"]
            gt_quality = targets["predicted_quality"]
            predictions = self.model(inputs, training=False)
            pred_heatmap, pred_quality = predictions
            
            if train_mode == "quality_only":
                loss_heatmap = tf.constant(0.0, dtype=tf.float32)
            else:
                loss_heatmap = coordinate_distance_loss(
                    gt_coords,
                    pred_heatmap,
                    gt_quality,
                    beta=self.config.get("soft_argmax_beta", 15.0)
                )
                
            if train_mode == "heatmap_only":
                loss_quality = tf.constant(0.0, dtype=tf.float32)
            else:
                pred_coords = get_peak_coords_tf(pred_heatmap)
                # Scale predicted coordinates to 256.0 space (from [0, H-1] space)
                H = tf.cast(tf.shape(pred_heatmap)[1], tf.float32)
                pred_coords_scaled = pred_coords * (256.0 / H)
                # Compare against gt_coords scaled to 256.0 space
                dist = tf.norm(pred_coords_scaled - gt_coords * 256.0, axis=-1, keepdims=True)
                dynamic_target = tf.maximum(1.0 - (dist / 30.0), 0.0)
                target_quality = tf.where(gt_quality > 0.5, dynamic_target, 0.0)
                target_quality = tf.stop_gradient(target_quality)
                loss_quality = loss_fn_quality(target_quality, pred_quality)
            
            if train_mode == "heatmap_only":
                loss_value = loss_heatmap
            elif train_mode == "quality_only":
                loss_value = loss_quality
            else:
                loss_value = loss_heatmap + 1.0 * loss_quality
            
            val_loss_avg.update_state(loss_value)
            val_hm_loss_avg.update_state(loss_heatmap)
            val_q_loss_avg.update_state(loss_quality)
            
            progress_bar.set_postfix(
                val_loss=float(val_loss_avg.result()),
                val_hm=float(val_hm_loss_avg.result()),
                val_q=float(val_q_loss_avg.result())
            )
            
        return float(val_loss_avg.result()), float(val_hm_loss_avg.result()), float(val_q_loss_avg.result())

    def train_epoch(self, dataset, optimizer, loss_fn_quality, epoch, num_epochs, train_mode="joint", steps=None, shuffle=True):
        if shuffle:
            buffer_size = steps if (steps is not None and steps > 0) else 100
            buffer_size = min(32, buffer_size)
            dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_hm_loss_avg = tf.keras.metrics.Mean()
        epoch_q_loss_avg = tf.keras.metrics.Mean()
        import tqdm
        import time as _time
        progress_bar = tqdm.tqdm(dataset, desc=f"Epoch {epoch:03d}/{num_epochs:03d}", total=steps, leave=False)

        # Per-epoch timing accumulators
        t_data = 0.0  # time waiting for next batch from the dataset iterator (data loading / prefetch stall)
        t_step = 0.0  # time inside GradientTape forward+backward + apply_gradients

        _it = iter(progress_bar)
        _t0 = _time.perf_counter()
        while True:
            try:
                batch = next(_it)
            except StopIteration:
                break
            t_data += _time.perf_counter() - _t0  # iterator returned → batch was ready

            inputs, targets = batch
            gt_coords = targets["predicted_coords"]
            gt_quality = targets["predicted_quality"]

            _t1 = _time.perf_counter()
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                pred_heatmap, pred_quality = predictions
                
                if train_mode == "quality_only":
                    loss_heatmap = tf.constant(0.0, dtype=tf.float32)
                else:
                    loss_heatmap = coordinate_distance_loss(
                        gt_coords,
                        pred_heatmap,
                        gt_quality,
                        beta=self.config.get("soft_argmax_beta", 15.0)
                    )
                    
                if train_mode == "heatmap_only":
                    loss_quality = tf.constant(0.0, dtype=tf.float32)
                else:
                    pred_coords = get_peak_coords_tf(pred_heatmap)
                    # Scale predicted coordinates to 256.0 space (from [0, H-1] space)
                    H = tf.cast(tf.shape(pred_heatmap)[1], tf.float32)
                    pred_coords_scaled = pred_coords * (256.0 / H)
                    # Compare against gt_coords scaled to 256.0 space
                    dist = tf.norm(pred_coords_scaled - gt_coords * 256.0, axis=-1, keepdims=True)
                    dynamic_target = tf.maximum(1.0 - (dist / 30.0), 0.0)
                    target_quality = tf.where(gt_quality > 0.5, dynamic_target, 0.0)
                    target_quality = tf.stop_gradient(target_quality)
                    loss_quality = loss_fn_quality(target_quality, pred_quality)
                
                if train_mode == "heatmap_only":
                    loss_value = loss_heatmap
                elif train_mode == "quality_only":
                    loss_value = loss_quality
                else:
                    loss_value = loss_heatmap + 1.0 * loss_quality
                
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            t_step += _time.perf_counter() - _t1

            epoch_loss_avg.update_state(loss_value)
            epoch_hm_loss_avg.update_state(loss_heatmap)
            epoch_q_loss_avg.update_state(loss_quality)
            
            progress_bar.set_postfix(
                loss=float(epoch_loss_avg.result()),
                hm=float(epoch_hm_loss_avg.result()),
                q=float(epoch_q_loss_avg.result())
            )
            _t0 = _time.perf_counter()  # restart timer for the next data-load wait

        return float(epoch_loss_avg.result()), float(epoch_hm_loss_avg.result()), float(epoch_q_loss_avg.result()), t_data, t_step

    def train(self, train_dataset, val_dataset, lr, num_of_epochs, train_steps=None, val_steps=None, loss_quality="bce", train_mode="joint", output_path=None, best_train_loss_output=None, log_file=None):
        if loss_quality == "bce":
            loss_fn_quality = losses.BinaryCrossentropy()
        elif loss_quality == "mse":
            loss_fn_quality = losses.MeanSquaredError()
        elif loss_quality == "huber":
            loss_fn_quality = losses.Huber()
        elif loss_quality == "logcosh":
            loss_fn_quality = losses.LogCosh()
        else:
            raise ValueError(f"Unknown quality loss: {loss_quality}")
            
        # Freeze layers dynamically based on Stage 1 / Stage 2 Training Mode
        if train_mode == "heatmap_only":
            quality_layer_names = [
                "quality_hm_conv", "quality_hm_pool", "quality_fusion", 
                "quality_conv1", "quality_gn1", "quality_relu1", 
                "quality_conv2", "quality_gn2", "quality_relu2", 
                "quality_gap", "quality_fc1", "predicted_quality"
            ]
            for name in quality_layer_names:
                try:
                    self.model.get_layer(name).trainable = False
                except Exception:
                    pass
            self.log("   [STAGE 1 - HEATMAP ONLY] Quality branch layers frozen completely.", log_file)
        elif train_mode == "quality_only":
            quality_layer_names = [
                "quality_hm_conv", "quality_hm_pool", "quality_fusion", 
                "quality_conv1", "quality_gn1", "quality_relu1", 
                "quality_conv2", "quality_gn2", "quality_relu2", 
                "quality_gap", "quality_fc1", "predicted_quality"
            ]
            for layer in self.model.layers:
                if layer.name not in quality_layer_names:
                    layer.trainable = False
            self.log("   [STAGE 2 - QUALITY ONLY] Shared encoders and heatmap decoder layers frozen completely.", log_file)
            
        optimizer = optimizers.Adam(learning_rate=lr, jit_compile=False)
        
        self.log(f"Calculating initial validation loss (Mode: {train_mode})...", log_file)
        best_val_loss, init_val_hm, init_val_q = self.evaluate(val_dataset, loss_fn_quality, train_mode=train_mode, steps=val_steps)
        best_train_loss = float('inf')
        self.log(f"Initial Validation Loss: {best_val_loss:.6f} (HM: {init_val_hm:.6f}, Q: {init_val_q:.6f})", log_file)
        
        import time as _time
        import gc

        for epoch in range(1, num_of_epochs + 1):
            _t_epoch_start = _time.perf_counter()
            epoch_loss, epoch_hm, epoch_q, t_data, t_step = self.train_epoch(train_dataset, optimizer, loss_fn_quality, epoch, num_of_epochs, train_mode=train_mode, steps=train_steps)

            # --- Save best train (timed) ---
            t_save = 0.0
            if best_train_loss_output and epoch_loss < best_train_loss:
                _t_s = _time.perf_counter()
                self.log(f"   [TRAIN IMPROVEMENT] Train loss improved from {best_train_loss:.6f} to {epoch_loss:.6f}. Saving to {best_train_loss_output}", log_file)
                best_train_loss = epoch_loss
                if os.path.dirname(best_train_loss_output):
                    os.makedirs(os.path.dirname(best_train_loss_output), exist_ok=True)
                self.model.save(best_train_loss_output)
                if self.config.get("normalization_type", "group_norm") == "batch_norm":
                     base, ext = os.path.splitext(best_train_loss_output)
                     self.fold_and_save(f"{base}_fbn{ext}")
                t_save += _time.perf_counter() - _t_s

            # --- Validation (timed) ---
            _t_val_start = _time.perf_counter()
            val_loss, val_hm, val_q = self.evaluate(val_dataset, loss_fn_quality, train_mode=train_mode, steps=val_steps)
            t_val = _time.perf_counter() - _t_val_start

            self.log(f"Epoch {epoch:03d}/{num_of_epochs:03d} | Train Loss: {epoch_loss:.6f} (HM: {epoch_hm:.6f}, Q: {epoch_q:.6f}) | Val Loss: {val_loss:.6f} (HM: {val_hm:.6f}, Q: {val_q:.6f})", log_file)

            # --- Save best val (timed) ---
            if val_loss < best_val_loss:
                _t_s = _time.perf_counter()
                self.log(f"   [VAL IMPROVEMENT] Val loss improved from {best_val_loss:.6f} to {val_loss:.6f}. Saving model...", log_file)
                best_val_loss = val_loss
                if output_path:
                    if os.path.dirname(output_path):
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    self.model.save(output_path)
                    if self.config.get("normalization_type", "group_norm") == "batch_norm":
                        base, ext = os.path.splitext(output_path)
                        self.fold_and_save(f"{base}_fbn{ext}")
                        dir_name = os.path.dirname(output_path)
                        self.fold_and_save(os.path.join(dir_name, "tracker_model_fbn.keras"))
                t_save += _time.perf_counter() - _t_s

            # --- Timing breakdown → log file only ---
            t_epoch_total = _time.perf_counter() - _t_epoch_start
            t_overhead = max(0.0, t_epoch_total - t_step - t_data - t_val - t_save)
            if log_file:
                def _pct(t):
                    return 100.0 * t / t_epoch_total if t_epoch_total > 0.0 else 0.0
                with open(log_file, "a") as _lf:
                    _lf.write(
                        f"   [TIMING] Total: {t_epoch_total:.1f}s"
                        f" | TrainStep: {t_step:.1f}s ({_pct(t_step):.1f}%)"
                        f" | DataLoad: {t_data:.1f}s ({_pct(t_data):.1f}%)"
                        f" | Validation: {t_val:.1f}s ({_pct(t_val):.1f}%)"
                        f" | Save: {t_save:.1f}s ({_pct(t_save):.1f}%)"
                        f" | Overhead: {t_overhead:.1f}s ({_pct(t_overhead):.1f}%)\n"
                    )

            gc.collect()

# Backward-compatible alias for existing utility scripts.
TargetTrackerVer4 = TargetTrackerVer5

# =====================================================================
# Dataset Pipeline
# =====================================================================

def load_hdf5_dataset(h5_path, batch_size, ref_shape, search_shape, val_split=0.1, is_val=False, train_mode="joint", cache_mode="cache", stream_chunk_size=1024):
    import h5py
    import psutil
    import numpy as np
    import gc
    import tensorflow as tf

    if cache_mode not in ("streaming", "auto", "cache"):
        raise ValueError(f"Unknown cache_mode: {cache_mode}. Expected 'streaming', 'auto', or 'cache'.")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 dataset not found at {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        total_samples = f['reference_stack'].shape[0]
        h5_ref_shape = f['reference_stack'].shape[1:]
        h5_search_shape = f['search_frame'].shape[1:]
        sample_size_bytes = (
            np.prod(h5_ref_shape) * f['reference_stack'].dtype.itemsize +
            np.prod(h5_search_shape) * f['search_frame'].dtype.itemsize +
            np.prod(f['ground_truth_coords'].shape[1:]) * f['ground_truth_coords'].dtype.itemsize +
            np.prod(f['ground_truth_quality'].shape[1:]) * f['ground_truth_quality'].dtype.itemsize
        )

    indices = np.arange(total_samples)
    np.random.RandomState(42).shuffle(indices)

    val_size = int(total_samples * val_split)
    selected_indices = indices[:val_size] if is_val else indices[val_size:]
    num_selected = (len(selected_indices) // batch_size) * batch_size
    selected_indices = selected_indices[:num_selected]

    if len(selected_indices) == 0:
        raise ValueError(f"No samples remaining after filtering/balancing for train_mode={train_mode}, is_val={is_val}")

    mem = psutil.virtual_memory()
    total_expected_bytes = len(selected_indices) * sample_size_bytes
    disk_chunk_size = max(batch_size, int(stream_chunk_size))
    disk_chunk_size = (disk_chunk_size // batch_size) * batch_size
    if disk_chunk_size == 0:
        disk_chunk_size = batch_size

    selected_positions = np.full(total_samples, -1, dtype=np.int64)
    selected_positions[selected_indices] = np.arange(len(selected_indices), dtype=np.int64)
    selected_mask = selected_positions >= 0

    use_ram_cache = cache_mode == "cache" or (cache_mode == "auto" and total_expected_bytes < mem.available * 0.40)

    def _batch_generator_from_arrays(ref_data, search_data, coords_data, quality_data):
        local_indices = np.arange(len(ref_data))
        if not is_val:
            np.random.shuffle(local_indices)

        for start_idx in range(0, len(local_indices), batch_size):
            batch_idx = local_indices[start_idx:start_idx + batch_size]
            if len(batch_idx) != batch_size:
                continue
            yield (
                {"reference_stack": ref_data[batch_idx], "search_frame": search_data[batch_idx]},
                {"predicted_coords": coords_data[batch_idx], "predicted_quality": quality_data[batch_idx]}
            )

    if use_ram_cache:
        print(f"Caching dataset split to RAM via contiguous HDF5 reads ({len(selected_indices)} samples, {total_expected_bytes / (1024**3):.2f} GB)...")
        read_block_size = max(disk_chunk_size, batch_size)
        with h5py.File(h5_path, 'r') as f:
            ref_ds = f['reference_stack']
            search_ds = f['search_frame']
            coords_ds = f['ground_truth_coords']
            quality_ds = f['ground_truth_quality']

            ref_data = np.empty((len(selected_indices), *ref_ds.shape[1:]), dtype=ref_ds.dtype)
            search_data = np.empty((len(selected_indices), *search_ds.shape[1:]), dtype=search_ds.dtype)
            coords_data = np.empty((len(selected_indices), *coords_ds.shape[1:]), dtype=coords_ds.dtype)
            quality_data = np.empty((len(selected_indices), *quality_ds.shape[1:]), dtype=quality_ds.dtype)

            for start_idx in range(0, total_samples, read_block_size):
                end_idx = min(start_idx + read_block_size, total_samples)
                block_positions = selected_positions[start_idx:end_idx]
                keep = block_positions >= 0
                if not np.any(keep):
                    continue
                dst = block_positions[keep]
                ref_data[dst] = ref_ds[start_idx:end_idx][keep]
                search_data[dst] = search_ds[start_idx:end_idx][keep]
                coords_data[dst] = coords_ds[start_idx:end_idx][keep]
                quality_data[dst] = quality_ds[start_idx:end_idx][keep]

        print("Dataset split successfully cached to RAM.")

        def chunk_generator():
            yield from _batch_generator_from_arrays(ref_data, search_data, coords_data, quality_data)
    else:
        print(f"Dataset streaming from HDF5 with contiguous reads (requires {total_expected_bytes / (1024**3):.2f} GB if cached, available {mem.available / (1024**3):.2f} GB). Chunk size: {disk_chunk_size} samples.")

        def chunk_generator():
            carry = None
            with h5py.File(h5_path, 'r') as f:
                ref_ds = f['reference_stack']
                search_ds = f['search_frame']
                coords_ds = f['ground_truth_coords']
                quality_ds = f['ground_truth_quality']

                for start_idx in range(0, total_samples, disk_chunk_size):
                    end_idx = min(start_idx + disk_chunk_size, total_samples)
                    keep = selected_mask[start_idx:end_idx]
                    if not np.any(keep):
                        continue

                    ref_chunk = ref_ds[start_idx:end_idx][keep]
                    search_chunk = search_ds[start_idx:end_idx][keep]
                    coords_chunk = coords_ds[start_idx:end_idx][keep]
                    quality_chunk = quality_ds[start_idx:end_idx][keep]

                    if not is_val:
                        order = np.arange(len(ref_chunk))
                        np.random.shuffle(order)
                        ref_chunk = ref_chunk[order]
                        search_chunk = search_chunk[order]
                        coords_chunk = coords_chunk[order]
                        quality_chunk = quality_chunk[order]

                    if carry is not None:
                        ref_chunk = np.concatenate((carry[0], ref_chunk), axis=0)
                        search_chunk = np.concatenate((carry[1], search_chunk), axis=0)
                        coords_chunk = np.concatenate((carry[2], coords_chunk), axis=0)
                        quality_chunk = np.concatenate((carry[3], quality_chunk), axis=0)
                        carry = None

                    full_count = (len(ref_chunk) // batch_size) * batch_size
                    for batch_start in range(0, full_count, batch_size):
                        batch_end = batch_start + batch_size
                        yield (
                            {"reference_stack": ref_chunk[batch_start:batch_end], "search_frame": search_chunk[batch_start:batch_end]},
                            {"predicted_coords": coords_chunk[batch_start:batch_end], "predicted_quality": quality_chunk[batch_start:batch_end]}
                        )

                    if full_count < len(ref_chunk):
                        carry = (
                            ref_chunk[full_count:],
                            search_chunk[full_count:],
                            coords_chunk[full_count:],
                            quality_chunk[full_count:]
                        )

                    ref_chunk = None
                    search_chunk = None
                    coords_chunk = None
                    quality_chunk = None
                    gc.collect()

    output_signature = (
        {
            "reference_stack": tf.TensorSpec(shape=(None, *ref_shape), dtype=tf.float32),
            "search_frame": tf.TensorSpec(shape=(None, *search_shape), dtype=tf.float32)
        },
        {
            "predicted_coords": tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
            "predicted_quality": tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        }
    )

    ds = tf.data.Dataset.from_generator(chunk_generator, output_signature=output_signature)
    ds = ds.prefetch(1)

    gc.collect()
    return ds, len(selected_indices)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="TargetTrackerVer5 Training CLI")
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--dataset_dir", nargs="+", required=True, help="One or more paths to dataset directories containing dataset.h5")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--dataset_cache_mode", choices=["streaming", "auto", "cache"], default="cache", help="Dataset loading mode. cache preloads selected samples to RAM using contiguous reads; streaming reads contiguous HDF5 blocks per epoch; auto caches when it appears safe.")
    parser.add_argument("--dataset_stream_chunk_size", type=int, default=1024, help="Maximum HDF5 samples to read per streaming chunk before batching. Used only when dataset_cache_mode=streaming or auto falls back to streaming.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_of_epochs", type=int, default=10)
    parser.add_argument("--loss_quality", choices=["bce", "mse", "huber", "logcosh"], default="bce")
    parser.add_argument("--train_mode", choices=["joint", "heatmap_only", "quality_only"], default="joint", help="Training mode: heatmap_only, quality_only, or joint")
    parser.add_argument("--output", type=str, default="outputs/tracker.keras")
    parser.add_argument("--best_train_loss_output", type=str, default="outputs/tracker_best_train.keras")
    parser.add_argument("--init_keras_file", type=str, default=None, help="Path to initial model to resume from")
    parser.add_argument("--log_file", type=str, default="outputs/train.log")
    parser.add_argument("--runtime_config_output", type=str, default="outputs/tracker_runtime_config.json", help="Path to write runtime preprocessing/shapes config JSON")
    
    args = parser.parse_args()
    
    if args.command == "train":
        check_and_create_default_config("model.conf")
        
        # Find dataset.h5 in specified directories
        h5_path = None
        for d in args.dataset_dir:
            candidate = os.path.join(d, "dataset.h5")
            if os.path.exists(candidate):
                h5_path = candidate
                break
        if not h5_path:
            raise FileNotFoundError(f"dataset.h5 not found in any of the specified directories: {args.dataset_dir}")
            
        print(f"Loading dataset from: {h5_path}")

        # Create tracker first so ref_shape / search_shape are derived from model.conf
        tracker = TargetTrackerVer5()
        print("Building TargetTrackerVer5 model...")
        tracker.create_model()

        train_ds, train_samples = load_hdf5_dataset(h5_path, batch_size=args.batch_size, ref_shape=tracker.ref_shape, search_shape=tracker.search_shape, val_split=args.val_split, is_val=False, train_mode=args.train_mode, cache_mode=args.dataset_cache_mode, stream_chunk_size=args.dataset_stream_chunk_size)
        val_ds, val_samples = load_hdf5_dataset(h5_path, batch_size=args.batch_size, ref_shape=tracker.ref_shape, search_shape=tracker.search_shape, val_split=args.val_split, is_val=True, train_mode=args.train_mode, cache_mode=args.dataset_cache_mode, stream_chunk_size=args.dataset_stream_chunk_size)

        train_steps = train_samples // args.batch_size
        val_steps = val_samples // args.batch_size

        print(f"Train samples: {train_samples} ({train_steps} steps/epoch)")
        print(f"Val samples: {val_samples} ({val_steps} steps/epoch)")

        if args.init_keras_file and os.path.exists(args.init_keras_file):
            print(f"Resuming training: loading weights from {args.init_keras_file}...")
            tracker.model.load_weights(args.init_keras_file, by_name=True, skip_mismatch=True)

        tracker.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            lr=args.lr,
            num_of_epochs=args.num_of_epochs,
            train_steps=train_steps,
            val_steps=val_steps,
            loss_quality=args.loss_quality,
            train_mode=args.train_mode,
            output_path=args.output,
            best_train_loss_output=args.best_train_loss_output,
            log_file=args.log_file
        )

        if args.runtime_config_output:
            tracker.save_runtime_config(args.runtime_config_output)


if __name__ == '__main__':
    main()
