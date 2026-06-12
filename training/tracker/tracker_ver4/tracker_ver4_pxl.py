import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import os
import pickle
import numpy as np
import configparser
from tqdm.auto import tqdm

# Global Normalization setting for dynamic layer configuration
_NORMALIZATION_TYPE = "group_norm"
_OVERRIDE_USE_BIAS = False

_orig_conv2d = layers.Conv2D
_orig_dw_conv2d = layers.DepthwiseConv2D

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

layers.Conv2D = _smart_conv2d
layers.DepthwiseConv2D = _smart_dw_conv2d

# Save the original GroupNormalization
if hasattr(layers, 'GroupNormalization'):
    GroupNormClass = layers.GroupNormalization
else:
    try:
        import tensorflow_addons as tfa
        GroupNormClass = tfa.layers.GroupNormalization
    except ImportError:
        class GroupNormClass(layers.Layer):
            def __init__(self, groups=8, **kwargs):
                super().__init__(**kwargs)
                self.groups = groups

@tf.keras.utils.register_keras_serializable(package="Custom")
class SafeGroupNormalization(GroupNormClass):
    def __init__(self, groups=8, **kwargs):
        self.requested_groups = groups
        super().__init__(groups=groups, **kwargs)
        
    def build(self, input_shape):
        channels = input_shape[-1]
        g = self.requested_groups
        if channels is not None and g is not None:
            if channels % g != 0:
                for candidate in range(self.requested_groups, 0, -1):
                    if channels % candidate == 0:
                        g = candidate
                        break
        self.groups = g
        super().build(input_shape)

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
        return SafeGroupNormalization(groups=g, name=name, **kwargs)

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

@tf.keras.utils.register_keras_serializable(package="Custom")
class HeatmapNormalization(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        return inputs / (tf.reduce_max(inputs, axis=[1, 2], keepdims=True) + 1e-7)

def load_model_config(config_path="model.conf"):
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # In this version, we share the backbone, so search_backbone defines the shared backbone
    shared_backbone = config.get("Backbone", "search_backbone", fallback="mnv2")
    width_mult = config.getfloat("Backbone", "width_multiplier", fallback=1.0)
    attn_mech = config.get("Attention", "mechanism", fallback="multi_head_cross")
    dec_type = config.get("Decoder", "type", fallback="unet")
    hm_loss_default = config.get("Loss", "heatmap_loss_default", fallback="adaptive_wing")
    norm_type = config.get("Normalization", "type", fallback="group_norm")
    
    return {
        "shared_backbone": shared_backbone,
        "width_multiplier": width_mult,
        "attention_mechanism": attn_mech,
        "decoder_type": dec_type,
        "heatmap_loss_default": hm_loss_default,
        "normalization_type": norm_type
    }

def check_and_create_default_config(config_path="model.conf"):
    if not os.path.exists(config_path):
        default_content = """# Google Siamese Subject Tracker (tracker_ver_pxl) Configuration File
# This file configures the shared backbone, attention fusion, and decoder.

[Backbone]
# Shared Siamese Backbone
# Options: mnv2 (Full MobileNetV2), mnv2_nano, mnv1, yolo5, custom_legacy
search_backbone = mnv2
width_multiplier = 1.0

[Attention]
# Fusion Attention Mechanism
# Options: depthwise_corr, dot_cross, linear_cross, multi_head_cross
mechanism = multi_head_cross

[Decoder]
# Heatmap Decoder Architecture
# Options: unet, fpn_add, pixel_shuffle, light_naive
type = unet

[Loss]
# Default loss function for heatmap regression
# Options: adaptive_wing, dbsz_relu, dbsz_soft, dbsz_hard, centernet_dice, focal_dice, mse
heatmap_loss_default = adaptive_wing

[Normalization]
# Normalization Type
# Options: group_norm, batch_norm
type = group_norm
"""
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(default_content)
        print(f"\n[CONFIG NOTICE] Configuration file '{config_path}' was not found. Created default config.")

# =====================================================================
# Custom Losses for Continuous Heatmap Regression
# =====================================================================

@tf.keras.utils.register_keras_serializable(package="Custom")
def dbsz_hard_loss(y_true, y_pred, c_bg=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask_high = tf.cast(y_true >= 0.5, tf.float32)
    mask_low = tf.cast(y_true <= 0.01, tf.float32)
    n_high = tf.reduce_sum(mask_high) + 1e-5
    n_low = tf.reduce_sum(mask_low) + 1e-5
    loss_high = tf.reduce_sum(mask_high * tf.abs(y_true - y_pred)) / n_high
    loss_low = tf.reduce_sum(mask_low * tf.square(y_true - y_pred)) / n_low
    return loss_high + c_bg * loss_low

@tf.keras.utils.register_keras_serializable(package="Custom")
def dbsz_soft_loss(y_true, y_pred, c_bg=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    w_high = tf.pow(y_true, 4.0)
    w_low = tf.pow(1.0 - y_true, 16.0)
    loss_high = tf.reduce_sum(w_high * tf.abs(y_true - y_pred)) / (tf.reduce_sum(w_high) + 1e-5)
    loss_low = tf.reduce_sum(w_low * tf.square(y_true - y_pred)) / (tf.reduce_sum(w_low) + 1e-5)
    return loss_high + c_bg * loss_low

@tf.keras.utils.register_keras_serializable(package="Custom")
def dbsz_relu_loss(y_true, y_pred, border=0.35, c_bg=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    w_high = tf.nn.relu(y_true - border)
    w_low = tf.nn.relu(border - y_true)
    k1 = 1.0 / (tf.reduce_sum(w_high) + 1e-5)
    k = 1.0 / (tf.reduce_sum(w_low) + 1e-5)
    loss_high = k1 * tf.reduce_sum(w_high * tf.square(y_true - y_pred))
    loss_low = k * tf.reduce_sum(w_low * tf.square(y_true - y_pred))
    return loss_high + c_bg * loss_low

@tf.keras.utils.register_keras_serializable(package="Custom")
def adaptive_wing_loss(y_true, y_pred, alpha=2.1, omega=14.0, epsilon=1.0, theta=0.5):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff = tf.abs(y_true - y_pred)
    exponent = alpha - y_true
    diff_clipped = tf.maximum(diff, 1e-7)
    ratio = diff_clipped / epsilon
    ratio_pow = tf.pow(ratio, exponent)
    loss_small = omega * tf.math.log1p(ratio_pow)
    theta_ratio = theta / epsilon
    theta_ratio_pow = tf.pow(theta_ratio, exponent)
    theta_ratio_pow_minus_1 = tf.pow(theta_ratio, exponent - 1.0)
    s = (omega / epsilon) * exponent * theta_ratio_pow_minus_1 / (1.0 + theta_ratio_pow)
    C = s * theta - omega * tf.math.log1p(theta_ratio_pow)
    loss_large = s * diff - C
    loss = tf.where(diff < theta, loss_small, loss_large)
    return tf.reduce_mean(loss)

@tf.keras.utils.register_keras_serializable(package="Custom")
def dice_bce_loss(y_true, y_pred, bce_weight=1.0, dice_weight=1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce_loss = tf.reduce_mean(bce)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    eps = 1e-12
    dice_coef = (2. * intersection + eps) / (union + eps)
    dice_loss = 1.0 - dice_coef
    return bce_weight * bce_loss + dice_weight * dice_loss

@tf.keras.utils.register_keras_serializable(package="Custom")
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    eps = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    bce_pos = - y_true * tf.math.log(y_pred) * tf.math.pow(1.0 - y_pred, gamma)
    bce_neg = - (1.0 - y_true) * tf.math.log(1.0 - y_pred) * tf.math.pow(y_pred, gamma)
    loss = alpha * bce_pos + (1.0 - alpha) * bce_neg
    return tf.reduce_mean(loss)

@tf.keras.utils.register_keras_serializable(package="Custom")
def focal_dice_loss(y_true, y_pred, focal_weight=1.0, dice_weight=1.0, alpha=0.25, gamma=2.0):
    eps = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    bce_pos = - y_true * tf.math.log(y_pred) * tf.math.pow(1.0 - y_pred, gamma)
    bce_neg = - (1.0 - y_true) * tf.math.log(1.0 - y_pred) * tf.math.pow(y_pred, gamma)
    focal_loss_val = tf.reduce_mean(alpha * bce_pos + (1.0 - alpha) * bce_neg)
    
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(tf.square(y_true_f)) + tf.reduce_sum(tf.square(y_pred_f))
    dice_loss_val = 1.0 - (2.0 * intersection + eps) / (denominator + eps)
    return focal_weight * focal_loss_val + dice_weight * dice_loss_val

@tf.keras.utils.register_keras_serializable(package="Custom")
def centernet_focal_loss(y_true, y_pred, alpha=2.0, beta=4.0):
    eps = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    pos_loss = - y_true * tf.math.pow(1.0 - y_pred, alpha) * tf.math.log(y_pred)
    neg_loss = - tf.math.pow(1.0 - y_true, beta) * tf.math.pow(y_pred, alpha) * tf.math.log(1.0 - y_pred)
    loss = pos_loss + neg_loss
    return tf.reduce_mean(loss)

@tf.keras.utils.register_keras_serializable(package="Custom")
def centernet_dice_loss(y_true, y_pred, focal_weight=1.0, dice_weight=1.0, alpha=2.0, beta=4.0):
    eps = 1e-7
    cn_loss = centernet_focal_loss(y_true, y_pred, alpha=alpha, beta=beta)
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(tf.square(y_true_f)) + tf.reduce_sum(tf.square(y_pred_f))
    dice_loss_val = 1.0 - (2.0 * intersection + eps) / (denominator + eps)
    return focal_weight * cn_loss + dice_weight * dice_loss_val

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
# Target Tracker Ver Pixel Class
# =====================================================================

class TargetTrackerVerPixel:
    def __init__(self, ref_shape=(64, 64, 16), search_shape=(256, 256, 1), config_path="model.conf"):
        self.ref_shape = ref_shape
        self.search_shape = search_shape
        self.model = None
        
        # Load configuration
        if os.path.exists(config_path):
            self.config = load_model_config(config_path)
        else:
            self.config = {
                "shared_backbone": "mnv2",
                "width_multiplier": 1.0,
                "attention_mechanism": "multi_head_cross",
                "decoder_type": "unet",
                "heatmap_loss_default": "adaptive_wing",
                "normalization_type": "group_norm"
            }

    def _inverted_residual_block(self, inputs, expansion, filters, strides, name_prefix):
        x = inputs
        in_channels = inputs.shape[-1]
        
        if expansion > 1:
            x = layers.Conv2D(expansion * in_channels, (1, 1), padding="same", use_bias=False, name=f"{name_prefix}_expand")(x)
            x = _GroupNormalization(expansion * in_channels, name=f"{name_prefix}_expand_gn")(x)
            x = layers.ReLU(6.0, name=f"{name_prefix}_expand_relu")(x)
            
        x = layers.DepthwiseConv2D((3, 3), strides=strides, padding="same", use_bias=False, name=f"{name_prefix}_dw")(x)
        x = _GroupNormalization(expansion * in_channels, name=f"{name_prefix}_dw_gn")(x)
        x = layers.ReLU(6.0, name=f"{name_prefix}_dw_relu")(x)
        
        x = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False, name=f"{name_prefix}_project")(x)
        x = _GroupNormalization(filters, name=f"{name_prefix}_project_gn")(x)
        
        if strides == 1 and in_channels == filters:
            x = layers.Add(name=f"{name_prefix}_add")([inputs, x])
            
        return x

    def _create_shared_backbone(self):
        """
        Creates the shared Siamese CNN backbone.
        This FCN backbone accepts dynamic input shapes (None, None, 1) and downsamples by 16x.
        It returns [x1, x2, x3, x] to support skip connections in the UNet/FPN decoders.
        """
        inputs = layers.Input(shape=(None, None, 1), name="shared_backbone_input")
        
        bb_type = self.config["shared_backbone"]
        width_mult = self.config["width_multiplier"]
        
        def scale_filters(f):
            return max(8, int(f * width_mult))
            
        if bb_type == "alex_net":
            # 5 layers of Conv2D + GroupNorm + ReLU
            # Strides: 2, 2, 2, 2, 1
            # Padding: same, so we get 128x128, 64x64, 32x32, 16x16, 16x16 (on 256x256 input)
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
            
        elif bb_type == "mnv2_nano":
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
            
            # Final Expand
            x = layers.Conv2D(scale_filters(128), (1, 1), padding="same", use_bias=False, name="sb_final_conv")(x4)
            x = _GroupNormalization(scale_filters(128), name="sb_final_gn")(x)
            x = layers.ReLU(6.0, name="sb_final_relu")(x)
            
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
            
        return models.Model(inputs, [x1, x2, x3, x], name="shared_siamese_backbone")

    def create_model(self):
        global _NORMALIZATION_TYPE, _OVERRIDE_USE_BIAS
        _NORMALIZATION_TYPE = self.config.get("normalization_type", "group_norm")
        _OVERRIDE_USE_BIAS = True
        try:
            return self._create_model_impl()
        finally:
            _OVERRIDE_USE_BIAS = False

    def _create_model_impl(self):
        ref_input = layers.Input(shape=self.ref_shape, name="reference_stack") # (1, 64, 64, 16)
        search_input = layers.Input(shape=self.search_shape, name="search_frame") # (256, 256, 1)
        
        # 1. Shared Siamese Backbone
        shared_backbone = self._create_shared_backbone()
        
        # 2. Process Reference Stack (Template)
        if len(self.ref_shape) == 3:
            # Native 4D format: (B, 64, 64, 16) -> transpose to (B, 16, 64, 64) -> reshape to (B*16, 64, 64, 1)
            ref_transposed = tf.transpose(ref_input, perm=[0, 3, 1, 2])
            ref_reshaped = tf.reshape(ref_transposed, (-1, self.ref_shape[0], self.ref_shape[1], 1))
        elif self.ref_shape[0] == 1:
            # New format: (B, 1, 64, 64, 16) -> transpose to (B, 16, 64, 64, 1) -> reshape to (B*16, 64, 64, 1)
            ref_transposed = tf.transpose(ref_input, perm=[0, 4, 2, 3, 1])
            ref_reshaped = tf.reshape(ref_transposed, (-1, self.ref_shape[1], self.ref_shape[2], 1))
        else:
            # Legacy format: (B, 16, 32, 32, 1) -> reshape to (B*16, 32, 32, 1)
            ref_reshaped = tf.reshape(ref_input, (-1, self.ref_shape[1], self.ref_shape[2], self.ref_shape[3]))
            
        ref_resized = tf.image.resize(ref_reshaped, (128, 128))
        
        ref_outputs = shared_backbone(ref_resized)
        ref_final_features = ref_outputs[-1] # Shape: (None * 16, 8, 8, C)
        
        if len(self.ref_shape) == 3:
            num_layers = self.ref_shape[2]
        else:
            num_layers = self.ref_shape[0] if self.ref_shape[0] > 1 else self.ref_shape[3]
        ref_features_split = tf.reshape(ref_final_features, (-1, num_layers, 8, 8, ref_final_features.shape[-1]))
        ref_features = tf.reduce_mean(ref_features_split, axis=1)
        
        # 3. Process Search Frame
        sb_init, sb_ir1, sb_ir2, search_features = shared_backbone(search_input)
        
        width_mult = self.config["width_multiplier"]
        def scale_filters(f):
            return max(8, int(f * width_mult))
            
        attn_mech = self.config["attention_mechanism"]
        
        # 4. Attention Fusion
        if attn_mech == "depthwise_corr":
            fused_features = DepthwiseCorrelationFusion(name="depthwise_correlation_fusion")([search_features, ref_features])
            c_v = scale_filters(128)
            
        elif attn_mech == "linear_cross":
            c_qk = scale_filters(64)
            c_v = scale_filters(128)
            
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
            c_v = scale_filters(128)
            q_flat = layers.Reshape((256, c_v), name="q_flat")(search_features)
            kv_flat = layers.Reshape((64, c_v), name="kv_flat")(ref_features)
            
            mha = layers.MultiHeadAttention(num_heads=4, key_dim=c_v // 4, name="mha_fusion")
            fused_flat = mha(query=q_flat, value=kv_flat, key=kv_flat)
            fused_features = layers.Reshape((16, 16, c_v), name="fused_features_reshape")(fused_flat)
            
        else: # dot_cross
            c_qk = scale_filters(64)
            c_v = scale_filters(128)
            
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
            
        # 5. Decoder for Output 1: Heatmap
        dec_type = self.config["decoder_type"]
        
        if dec_type == "fpn_add":
            x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_up1")(fused_features)
            skip1_proj = layers.Conv2D(x.shape[-1], (1, 1), padding="same", use_bias=False, name="decoder_skip1_proj")(sb_ir2)
            x = layers.Add(name="decoder_skip1")([x, skip1_proj])
            x = self._inverted_residual_block(x, expansion=2, filters=scale_filters(64), strides=1, name_prefix="decoder_ir1")
            
            x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_up2")(x)
            skip2_proj = layers.Conv2D(x.shape[-1], (1, 1), padding="same", use_bias=False, name="decoder_skip2_proj")(sb_ir1)
            x = layers.Add(name="decoder_skip2")([x, skip2_proj])
            x = self._inverted_residual_block(x, expansion=2, filters=scale_filters(32), strides=1, name_prefix="decoder_ir2")
            
            x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_up3")(x)
            skip3_proj = layers.Conv2D(x.shape[-1], (1, 1), padding="same", use_bias=False, name="decoder_skip3_proj")(sb_init)
            x = layers.Add(name="decoder_skip3")([x, skip3_proj])
            x = self._inverted_residual_block(x, expansion=2, filters=scale_filters(16), strides=1, name_prefix="decoder_ir3")
            
            x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_up4")(x)
            x = self._inverted_residual_block(x, expansion=2, filters=scale_filters(8), strides=1, name_prefix="decoder_ir4")
            
        elif dec_type == "pixel_shuffle":
            def pixel_shuffle_block(inputs, out_filters, name_prefix):
                ps_c = out_filters * 4
                x_ps = layers.Conv2D(ps_c, (3, 3), padding="same", use_bias=False, name=f"{name_prefix}_ps_conv")(inputs)
                x_ps = _GroupNormalization(ps_c, name=f"{name_prefix}_ps_gn")(x_ps)
                x_ps = layers.ReLU(6.0, name=f"{name_prefix}_ps_relu")(x_ps)
                x_ps = DepthToSpace(block_size=2, name=f"{name_prefix}_ps_shuffle")(x_ps)
                return x_ps
                
            x = pixel_shuffle_block(fused_features, scale_filters(64), "decoder1")
            skip1_proj = layers.Conv2D(x.shape[-1], (1, 1), padding="same", use_bias=False, name="decoder_skip1_proj")(sb_ir2)
            x = layers.Add(name="decoder_skip1")([x, skip1_proj])
            
            x = pixel_shuffle_block(x, scale_filters(32), "decoder2")
            skip2_proj = layers.Conv2D(x.shape[-1], (1, 1), padding="same", use_bias=False, name="decoder_skip2_proj")(sb_ir1)
            x = layers.Add(name="decoder_skip2")([x, skip2_proj])
            
            x = pixel_shuffle_block(x, scale_filters(16), "decoder3")
            skip3_proj = layers.Conv2D(x.shape[-1], (1, 1), padding="same", use_bias=False, name="decoder_skip3_proj")(sb_init)
            x = layers.Add(name="decoder_skip3")([x, skip3_proj])
            
            x = pixel_shuffle_block(x, scale_filters(8), "decoder4")
            
        elif dec_type == "light_naive":
            c_up1 = scale_filters(64)
            x = layers.Conv2DTranspose(c_up1, (3, 3), strides=2, padding="same", use_bias=False, name="decoder_up1")(fused_features)
            x = _GroupNormalization(c_up1, name="decoder_up1_gn")(x)
            x = layers.ReLU(6.0, name="decoder_up1_relu")(x)
            
            c_up2 = scale_filters(32)
            x = layers.Conv2DTranspose(c_up2, (3, 3), strides=2, padding="same", use_bias=False, name="decoder_up2")(x)
            x = _GroupNormalization(c_up2, name="decoder_up2_gn")(x)
            x = layers.ReLU(6.0, name="decoder_up2_relu")(x)
            
            c_up3 = scale_filters(16)
            x = layers.Conv2DTranspose(c_up3, (3, 3), strides=2, padding="same", use_bias=False, name="decoder_up3")(x)
            x = _GroupNormalization(c_up3, name="decoder_up3_gn")(x)
            x = layers.ReLU(6.0, name="decoder_up3_relu")(x)
            
            c_up4 = scale_filters(8)
            x = layers.Conv2DTranspose(c_up4, (3, 3), strides=2, padding="same", use_bias=False, name="decoder_up4")(x)
            x = _GroupNormalization(c_up4, name="decoder_up4_gn")(x)
            x = layers.ReLU(6.0, name="decoder_up4_relu")(x)
            
        else: # unet
            x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_up1")(fused_features)
            x = layers.Concatenate(axis=-1, name="decoder_skip1")([x, sb_ir2])
            x = self._inverted_residual_block(x, expansion=2, filters=scale_filters(64), strides=1, name_prefix="decoder_ir1")
            
            x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_up2")(x)
            x = layers.Concatenate(axis=-1, name="decoder_skip2")([x, sb_ir1])
            x = self._inverted_residual_block(x, expansion=2, filters=scale_filters(32), strides=1, name_prefix="decoder_ir2")
            
            x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_up3")(x)
            x = layers.Concatenate(axis=-1, name="decoder_skip3")([x, sb_init])
            x = self._inverted_residual_block(x, expansion=2, filters=scale_filters(16), strides=1, name_prefix="decoder_ir3")
            
            x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_up4")(x)
            x = self._inverted_residual_block(x, expansion=2, filters=scale_filters(8), strides=1, name_prefix="decoder_ir4")
            
        # Final prediction heatmap
        output_heatmap_raw = layers.Conv2D(1, (3, 3), padding="same", activation="relu", name="predicted_heatmap_raw")(x)
        output_heatmap = HeatmapNormalization(name="predicted_heatmap")(output_heatmap_raw)
        
        # 6. Heatmap-Guided Classification Branch
        hm_feat = layers.Conv2D(8, (3, 3), strides=2, padding="same", activation="relu", name="quality_hm_conv")(output_heatmap)
        hm_pool = layers.AveragePooling2D(pool_size=8, name="quality_hm_pool")(hm_feat)
        
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
            name="TargetTrackerVerPixel"
        )
        
        print("\n" + "="*50)
        print("Model Architectural Configuration (Google Siamese):")
        for k, v in self.config.items():
            print(f"  {k}: {v}")
        print("="*50)
        self.model.summary()
        print(f"Total Model Parameters: {self.model.count_params():,}")
        print("="*50 + "\n")
        
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

    def log(self, message, log_file=None):
        import tqdm
        tqdm.tqdm.write(message)
        if log_file:
            parent_dir = os.path.dirname(log_file)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(message + "\n")

    def evaluate(self, dataset, loss_fn_heatmap, loss_fn_quality, train_mode="joint", steps=None):
        val_loss_avg = tf.keras.metrics.Mean()
        val_hm_loss_avg = tf.keras.metrics.Mean()
        val_q_loss_avg = tf.keras.metrics.Mean()
        import tqdm
        
        progress_bar = tqdm.tqdm(dataset, desc="Evaluating", total=steps, leave=False)
        for inputs, targets in progress_bar:
            gt_heatmap = targets["predicted_heatmap"]
            gt_quality = targets["predicted_quality"]
            predictions = self.model(inputs, training=False)
            pred_heatmap, pred_quality = predictions
            
            if train_mode == "quality_only":
                loss_heatmap = tf.constant(0.0, dtype=tf.float32)
            else:
                loss_heatmap = loss_fn_heatmap(gt_heatmap, pred_heatmap)
                
            loss_quality = loss_fn_quality(gt_quality, pred_quality)
            
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

    def train_epoch(self, dataset, optimizer, loss_fn_heatmap, loss_fn_quality, epoch, num_epochs, train_mode="joint", steps=None, shuffle=True):
        if shuffle:
            buffer_size = steps if (steps is not None and steps > 0) else 100
            buffer_size = min(32, buffer_size)
            dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_hm_loss_avg = tf.keras.metrics.Mean()
        epoch_q_loss_avg = tf.keras.metrics.Mean()
        import tqdm
        progress_bar = tqdm.tqdm(dataset, desc=f"Epoch {epoch:03d}/{num_epochs:03d}", total=steps, leave=False)
        
        for inputs, targets in progress_bar:
            gt_heatmap = targets["predicted_heatmap"]
            gt_quality = targets["predicted_quality"]
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                pred_heatmap, pred_quality = predictions
                
                if train_mode == "quality_only":
                    loss_heatmap = tf.constant(0.0, dtype=tf.float32)
                else:
                    loss_heatmap = loss_fn_heatmap(gt_heatmap, pred_heatmap)
                    
                loss_quality = loss_fn_quality(gt_quality, pred_quality)
                
                if train_mode == "heatmap_only":
                    loss_value = loss_heatmap
                elif train_mode == "quality_only":
                    loss_value = loss_quality
                else:
                    loss_value = loss_heatmap + 1.0 * loss_quality
                
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            epoch_loss_avg.update_state(loss_value)
            epoch_hm_loss_avg.update_state(loss_heatmap)
            epoch_q_loss_avg.update_state(loss_quality)
            
            progress_bar.set_postfix(
                loss=float(epoch_loss_avg.result()),
                hm=float(epoch_hm_loss_avg.result()),
                q=float(epoch_q_loss_avg.result())
            )
            
        return float(epoch_loss_avg.result()), float(epoch_hm_loss_avg.result()), float(epoch_q_loss_avg.result())

    def train(self, train_dataset, val_dataset, lr, num_of_epochs, train_steps=None, val_steps=None, loss_heatmap="adaptive_wing", loss_quality="bce", train_mode="joint", c_bg=3.0, dbsz_border=0.35, output_path=None, best_train_loss_output=None, log_file=None):
        if loss_heatmap == "mse":
            loss_fn_heatmap = losses.MeanSquaredError()
        elif loss_heatmap == "dice_bce":
            loss_fn_heatmap = dice_bce_loss
        elif loss_heatmap == "focal":
            loss_fn_heatmap = focal_loss
        elif loss_heatmap == "focal_dice":
            loss_fn_heatmap = focal_dice_loss
        elif loss_heatmap == "centernet":
            loss_fn_heatmap = centernet_focal_loss
        elif loss_heatmap == "centernet_dice":
            loss_fn_heatmap = centernet_dice_loss
        elif loss_heatmap == "adaptive_wing":
            loss_fn_heatmap = adaptive_wing_loss
        elif loss_heatmap == "dbsz_hard":
            loss_fn_heatmap = lambda y_true, y_pred: dbsz_hard_loss(y_true, y_pred, c_bg=c_bg)
        elif loss_heatmap == "dbsz_soft":
            loss_fn_heatmap = lambda y_true, y_pred: dbsz_soft_loss(y_true, y_pred, c_bg=c_bg)
        elif loss_heatmap == "dbsz_relu":
            loss_fn_heatmap = lambda y_true, y_pred: dbsz_relu_loss(y_true, y_pred, border=dbsz_border, c_bg=c_bg)
        else:
            raise ValueError(f"Unknown heatmap loss: {loss_heatmap}")
            
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
        best_val_loss, init_val_hm, init_val_q = self.evaluate(val_dataset, loss_fn_heatmap, loss_fn_quality, train_mode=train_mode, steps=val_steps)
        best_train_loss = float('inf')
        self.log(f"Initial Validation Loss: {best_val_loss:.6f} (HM: {init_val_hm:.6f}, Q: {init_val_q:.6f})", log_file)
        
        for epoch in range(1, num_of_epochs + 1):
            epoch_loss, epoch_hm, epoch_q = self.train_epoch(train_dataset, optimizer, loss_fn_heatmap, loss_fn_quality, epoch, num_of_epochs, train_mode=train_mode, steps=train_steps)
            
            if best_train_loss_output and epoch_loss < best_train_loss:
                self.log(f"   [TRAIN IMPROVEMENT] Train loss improved from {best_train_loss:.6f} to {epoch_loss:.6f}. Saving to {best_train_loss_output}", log_file)
                best_train_loss = epoch_loss
                if os.path.dirname(best_train_loss_output):
                    os.makedirs(os.path.dirname(best_train_loss_output), exist_ok=True)
                self.model.save(best_train_loss_output)
                if self.config.get("normalization_type", "group_norm") == "batch_norm":
                    base, ext = os.path.splitext(best_train_loss_output)
                    self.fold_and_save(f"{base}_fbn{ext}")
            
            val_loss, val_hm, val_q = self.evaluate(val_dataset, loss_fn_heatmap, loss_fn_quality, train_mode=train_mode, steps=val_steps)
            self.log(f"Epoch {epoch:03d}/{num_of_epochs:03d} | Train Loss: {epoch_loss:.6f} (HM: {epoch_hm:.6f}, Q: {epoch_q:.6f}) | Val Loss: {val_loss:.6f} (HM: {val_hm:.6f}, Q: {val_q:.6f})", log_file)
            
            if val_loss < best_val_loss:
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
                        self.fold_and_save(os.path.join(dir_name, "tracker_pxl_fbn.keras"))
                        self.fold_and_save(os.path.join(dir_name, "tracker_model_fbn.keras"))
            
            import gc
            gc.collect()

# =====================================================================
# Dataset Pipeline
# =====================================================================

def read_hdf5_sliced(dataset_obj, indices, chunk_size=10000):
    """
    Slices a large HDF5 dataset. If the dataset fits in less than 50% of available memory,
    we read the entire dataset contiguously to RAM and slice it in memory, bypassing
    h5py's extremely slow scattered index selection.
    """
    import numpy as np
    import psutil
    
    mem = psutil.virtual_memory()
    total_dataset_bytes = dataset_obj.shape[0] * np.prod(dataset_obj.shape[1:]) * 4
    
    if total_dataset_bytes < mem.available * 0.50:
        # Contiguous block read (very fast)
        data = dataset_obj[:]
        return data[indices]
        
    # Fallback to memory-safe sorted index chunks
    chunks = []
    for i in range(0, len(indices), chunk_size):
        batch_idx = indices[i:i + chunk_size]
        sort_idx = np.argsort(batch_idx)
        sorted_idx = batch_idx[sort_idx]
        
        chunk_data = dataset_obj[sorted_idx]
        
        unsort_idx = np.argsort(sort_idx)
        chunks.append(chunk_data[unsort_idx])
        
    return np.concatenate(chunks, axis=0)

def load_hdf5_dataset(h5_path, batch_size, val_split=0.1, is_val=False, train_mode="joint"):
    import h5py
    import psutil
    import numpy as np
    import gc
    import tensorflow as tf
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 dataset not found at {h5_path}")
        
    with h5py.File(h5_path, 'r') as f:
        total_samples = f['reference_stack'].shape[0]
        
    # Generate deterministic shuffled indices
    indices = np.arange(total_samples)
    np.random.RandomState(42).shuffle(indices)
    
    val_size = int(total_samples * val_split)
    if is_val:
        selected_indices = indices[:val_size]
    else:
        selected_indices = indices[val_size:]
    # Make divisible by batch_size to avoid partial batch tails
    num_selected = len(selected_indices)
    if num_selected % batch_size != 0:
        num_selected = (num_selected // batch_size) * batch_size
        selected_indices = selected_indices[:num_selected]
        
    if len(selected_indices) == 0:
        raise ValueError(f"No samples remaining after filtering/balancing for train_mode={train_mode}, is_val={is_val}")
        
    # Check RAM caching capability using psutil
    sample_size_bytes = (64 * 64 * 16 + 256 * 256 * 1 + 256 * 256 * 1) * 4 + 4
    total_expected_bytes = len(selected_indices) * sample_size_bytes
    
    mem = psutil.virtual_memory()
    
    # Calculate fallback chunk size for dynamic disk streaming
    target_chunk_bytes = mem.available * 0.15
    disk_chunk_size = int(target_chunk_bytes // sample_size_bytes)
    disk_chunk_size = max(2000, min(8000, disk_chunk_size))
    disk_chunk_size = (disk_chunk_size // batch_size) * batch_size
    if disk_chunk_size == 0:
        disk_chunk_size = batch_size
        
    # Cache to RAM only if it occupies less than 40% of currently available memory
    if total_expected_bytes < mem.available * 0.40:
        print(f"Caching dataset to RAM ({len(selected_indices)} samples, {total_expected_bytes / (1024**3):.2f} GB)...")
        with h5py.File(h5_path, 'r') as f:
            ref_data = read_hdf5_sliced(f['reference_stack'], selected_indices)
            search_data = read_hdf5_sliced(f['search_frame'], selected_indices)
            heatmap_data = read_hdf5_sliced(f['ground_truth_heatmap'], selected_indices)
            quality_data = read_hdf5_sliced(f['ground_truth_quality'], selected_indices)
            
        print("Dataset successfully cached to RAM.")
        
        # Generator yielding chunks from preloaded RAM arrays to prevent GPU OOM
        def chunk_generator():
            import math
            local_indices = np.arange(len(selected_indices))
            if not is_val:
                np.random.shuffle(local_indices)
                
            ram_chunk_size = 5000
            num_chunks = int(math.ceil(len(local_indices) / ram_chunk_size))
            for i in range(num_chunks):
                chunk_idx = local_indices[i * ram_chunk_size : (i + 1) * ram_chunk_size]
                yield (
                    ref_data[chunk_idx],
                    search_data[chunk_idx]
                ), (
                    heatmap_data[chunk_idx],
                    quality_data[chunk_idx]
                )
    else:
        # Load dynamically in memory-safe chunks directly from HDF5
        print(f"Dataset streaming from HDF5 (requires {total_expected_bytes / (1024**3):.2f} GB, available {mem.available / (1024**3):.2f} GB). Chunk size: {disk_chunk_size} samples.")
        
        def chunk_generator():
            import math
            local_indices = selected_indices.copy()
            if not is_val:
                np.random.shuffle(local_indices)
                
            num_chunks = int(math.ceil(len(local_indices) / disk_chunk_size))
            for i in range(num_chunks):
                chunk_idx = local_indices[i * disk_chunk_size : (i + 1) * disk_chunk_size]
                
                with h5py.File(h5_path, 'r') as f:
                    ref_chunk = read_hdf5_sliced(f['reference_stack'], chunk_idx)
                    search_chunk = read_hdf5_sliced(f['search_frame'], chunk_idx)
                    heatmap_chunk = read_hdf5_sliced(f['ground_truth_heatmap'], chunk_idx)
                    quality_chunk = read_hdf5_sliced(f['ground_truth_quality'], chunk_idx)
                    
                yield (ref_chunk, search_chunk), (heatmap_chunk, quality_chunk)
                
                ref_chunk = None
                search_chunk = None
                heatmap_chunk = None
                quality_chunk = None
                gc.collect()
                
    # Define output signature with dynamic leading dimension
    output_signature = (
        (
            tf.TensorSpec(shape=(None, 64, 64, 16), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32)
        ),
        (
            tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        )
    )
    
    ds = tf.data.Dataset.from_generator(chunk_generator, output_signature=output_signature)
    ds = ds.unbatch()
        
    # Map to format required by Keras models
    def process_element(inputs, targets):
        if isinstance(inputs, dict):
            return inputs, targets
        ref, search = inputs
        heatmap, quality = targets
        return {"reference_stack": ref, "search_frame": search}, {"predicted_heatmap": heatmap, "predicted_quality": quality}
        
    ds = ds.map(process_element, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    gc.collect()
    return ds, len(selected_indices)

def main():
    import argparse
    
    # We can pre-load config loss default if model.conf exists
    default_hm_loss = "adaptive_wing"
    if os.path.exists("model.conf"):
        try:
            cfg = load_model_config("model.conf")
            default_hm_loss = cfg.get("heatmap_loss_default", "adaptive_wing")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="TargetTrackerVerPixel Training CLI")
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--dataset_dir", nargs="+", required=True, help="One or more paths to dataset directories containing dataset.h5")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--dbsz_border", type=float, default=0.35, help="Border parameter for dbsz_relu_loss")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_of_epochs", type=int, default=10)
    parser.add_argument("--loss_heatmap", choices=["mse", "dice_bce", "focal", "focal_dice", "centernet", "centernet_dice", "adaptive_wing", "dbsz_hard", "dbsz_soft", "dbsz_relu"], default=default_hm_loss)
    parser.add_argument("--loss_quality", choices=["bce", "mse", "huber", "logcosh"], default="bce")
    parser.add_argument("--train_mode", choices=["joint", "heatmap_only", "quality_only"], default="joint")
    parser.add_argument("--c_bg", type=float, default=3.0, help="Background suppression weight factor for DBSZ losses")
    parser.add_argument("--output", type=str, default="outputs/tracker_pxl.keras")
    parser.add_argument("--best_train_loss_output", type=str, default="outputs/tracker_pxl_best_train.keras")
    parser.add_argument("--init_keras_file", type=str, default=None, help="Path to initial model to resume from")
    parser.add_argument("--log_file", type=str, default="outputs/train.log")
    
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
        
        train_ds, train_samples = load_hdf5_dataset(h5_path, batch_size=args.batch_size, val_split=args.val_split, is_val=False, train_mode=args.train_mode)
        val_ds, val_samples = load_hdf5_dataset(h5_path, batch_size=args.batch_size, val_split=args.val_split, is_val=True, train_mode=args.train_mode)
        
        train_steps = train_samples // args.batch_size
        val_steps = val_samples // args.batch_size
        
        print(f"Train samples: {train_samples} ({train_steps} steps/epoch)")
        print(f"Val samples: {val_samples} ({val_steps} steps/epoch)")
        
        tracker = TargetTrackerVerPixel()
        print("Building TargetTrackerVerPixel model...")
        tracker.create_model()
        
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
            loss_heatmap=args.loss_heatmap,
            loss_quality=args.loss_quality,
            train_mode=args.train_mode,
            c_bg=args.c_bg,
            dbsz_border=args.dbsz_border,
            output_path=args.output,
            best_train_loss_output=args.best_train_loss_output,
            log_file=args.log_file
        )

if __name__ == '__main__':
    main()
