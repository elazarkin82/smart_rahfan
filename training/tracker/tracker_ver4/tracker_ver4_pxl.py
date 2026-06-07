import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import os
import pickle
import numpy as np
import configparser

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
    g = get_safe_groups(channels)
    return GroupNormClass(groups=g, name=name, **kwargs)

@tf.keras.utils.register_keras_serializable(package="Custom")
class DepthwiseCorrelationFusion(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        search_feat, ref_feat = inputs
        def corr_single(x):
            s, r = x
            s = tf.expand_dims(s, 0)
            r = tf.expand_dims(r, -1)
            out = tf.nn.depthwise_conv2d(s, r, strides=[1, 1, 1, 1], padding="SAME")
            return out[0]
        return tf.map_fn(corr_single, [search_feat, ref_feat], fn_output_signature=tf.float32)

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
    
    return {
        "shared_backbone": shared_backbone,
        "width_multiplier": width_mult,
        "attention_mechanism": attn_mech,
        "decoder_type": dec_type,
        "heatmap_loss_default": hm_loss_default
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
def dbsz_relu_loss(y_true, y_pred, c_bg=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    w_high = tf.nn.relu(y_true - 0.5)
    w_low = tf.nn.relu(0.2 - y_true)
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
# Target Tracker Ver Pixel Class
# =====================================================================

class TargetTrackerVerPixel:
    def __init__(self, ref_shape=(16, 32, 32, 1), search_shape=(256, 256, 1), config_path="model.conf"):
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
                "heatmap_loss_default": "adaptive_wing"
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
            
        # 1. Stride 2 -> 1/2 size
        x1 = layers.Conv2D(scale_filters(16), (3, 3), strides=2, padding="same", use_bias=False, name="sb_init_conv")(inputs)
        x1 = _GroupNormalization(scale_filters(16), name="sb_init_gn")(x1)
        x1 = layers.ReLU(6.0, name="sb_init_relu")(x1)
        
        # 2. Stride 4 -> 1/4 size
        x2 = self._inverted_residual_block(x1, expansion=2, filters=scale_filters(16), strides=2, name_prefix="sb_ir1")
        
        # 3. Stride 8 -> 1/8 size
        x3 = self._inverted_residual_block(x2, expansion=2, filters=scale_filters(24), strides=2, name_prefix="sb_ir2")
        
        # 4. Stride 16 -> 1/16 size
        x4 = self._inverted_residual_block(x3, expansion=3, filters=scale_filters(32), strides=2, name_prefix="sb_ir3")
        
        # Final Expand
        x = layers.Conv2D(scale_filters(128), (1, 1), padding="same", use_bias=False, name="sb_final_conv")(x4)
        x = _GroupNormalization(scale_filters(128), name="sb_final_gn")(x)
        x = layers.ReLU(6.0, name="sb_final_relu")(x)
        
        return models.Model(inputs, [x1, x2, x3, x], name="shared_siamese_backbone")

    def create_model(self):
        ref_input = layers.Input(shape=self.ref_shape, name="reference_stack") # (16, 32, 32, 1)
        search_input = layers.Input(shape=self.search_shape, name="search_frame") # (256, 256, 1)
        
        # 1. Shared Siamese Backbone
        shared_backbone = self._create_shared_backbone()
        
        # 2. Process Reference Stack (Template)
        ref_reshaped = tf.reshape(ref_input, (-1, self.ref_shape[1], self.ref_shape[2], self.ref_shape[3]))
        ref_resized = tf.image.resize(ref_reshaped, (128, 128))
        
        ref_outputs = shared_backbone(ref_resized)
        ref_final_features = ref_outputs[-1] # Shape: (None * 16, 8, 8, C)
        
        ref_features_split = tf.reshape(ref_final_features, (-1, self.ref_shape[0], 8, 8, ref_final_features.shape[-1]))
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

    def train(self, train_dataset, val_dataset, lr, num_of_epochs, train_steps=None, val_steps=None, loss_heatmap="adaptive_wing", loss_quality="bce", train_mode="joint", c_bg=3.0, output_path=None, best_train_loss_output=None, log_file=None):
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
            loss_fn_heatmap = lambda y_true, y_pred: dbsz_relu_loss(y_true, y_pred, c_bg=c_bg)
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
            
            val_loss, val_hm, val_q = self.evaluate(val_dataset, loss_fn_heatmap, loss_fn_quality, train_mode=train_mode, steps=val_steps)
            self.log(f"Epoch {epoch:03d}/{num_of_epochs:03d} | Train Loss: {epoch_loss:.6f} (HM: {epoch_hm:.6f}, Q: {epoch_q:.6f}) | Val Loss: {val_loss:.6f} (HM: {val_hm:.6f}, Q: {val_q:.6f})", log_file)
            
            if val_loss < best_val_loss:
                self.log(f"   [VAL IMPROVEMENT] Val loss improved from {best_val_loss:.6f} to {val_loss:.6f}. Saving model...", log_file)
                best_val_loss = val_loss
                if output_path:
                    if os.path.dirname(output_path):
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    self.model.save(output_path)

# =====================================================================
# Dataset Pipeline
# =====================================================================

def parse_training_samples(pkl_path):
    with open(pkl_path, 'rb') as f:
        samples = pickle.load(f)
    
    refs = np.stack([s['reference_stack'] for s in samples], axis=0)
    searches = np.stack([s['search_frame'] for s in samples], axis=0)
    heatmaps = np.stack([s['ground_truth_heatmap'] for s in samples], axis=0)
    qualities = np.stack([s.get('ground_truth_quality', np.array([1.0], dtype=np.float16)) for s in samples], axis=0)
    
    return (refs, searches), (heatmaps, qualities)

def build_tf_dataset(pkl_files, shuffle=True):
    if not pkl_files:
        raise ValueError("No PKL files provided for dataset.")
        
    def generator():
        local_files = list(pkl_files)
        if shuffle:
            import random
            random.shuffle(local_files)
        for path in local_files:
            yield parse_training_samples(path)
            
    output_signature = (
        (
            tf.TensorSpec(shape=(None, 16, 32, 32, 1), dtype=tf.uint8, name="reference_stack"),
            tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.uint8, name="search_frame")
        ),
        (
            tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float16, name="heatmap"),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float16, name="quality")
        )
    )
    
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    
    def process_element(inputs, targets):
        ref, search = inputs
        heatmap, quality = targets
        
        ref_float = tf.cast(ref, tf.float32) / 255.0
        search_resized = tf.image.resize(search, (256, 256))
        search_float = tf.cast(search_resized, tf.float32) / 255.0
        heatmap_resized = tf.image.resize(tf.cast(heatmap, tf.float32), (256, 256))
        
        return {"reference_stack": ref_float, "search_frame": search_float}, {"predicted_heatmap": heatmap_resized, "predicted_quality": tf.cast(quality, tf.float32)}

    ds = ds.map(process_element, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

def main():
    import argparse
    
    default_hm_loss = "adaptive_wing"
    if os.path.exists("model.conf"):
        try:
            cfg = load_model_config("model.conf")
            default_hm_loss = cfg.get("heatmap_loss_default", "adaptive_wing")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="TargetTrackerVerPixel Training CLI")
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--dataset_dir", nargs="+", required=True, help="One or more paths to dataset PKL directories")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval_pkl_num", type=int, default=4, help="Number of PKLs for validation")
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
        import glob
        
        all_pkls = []
        for d in args.dataset_dir:
            all_pkls.extend(glob.glob(os.path.join(d, "batch_*.pkl")))
                
        all_pkls = sorted(all_pkls)
        if not all_pkls:
            raise FileNotFoundError(f"No batch files found in specified directories: {args.dataset_dir}")
            
        import random
        random.Random(42).shuffle(all_pkls)
            
        val_files = all_pkls[:args.eval_pkl_num]
        train_files = all_pkls[args.eval_pkl_num:]
        if not train_files:
            train_files = val_files
            
        train_ds = build_tf_dataset(train_files, shuffle=True)
        val_ds = build_tf_dataset(val_files, shuffle=False)
        
        tracker = TargetTrackerVerPixel()
        
        if args.init_keras_file and os.path.exists(args.init_keras_file):
            print(f"Resuming training: loading model from {args.init_keras_file}...")
            tracker.model = tf.keras.models.load_model(args.init_keras_file, compile=False, safe_mode=False)
        else:
            print("Building new TargetTrackerVerPixel model...")
            tracker.create_model()
            
        tracker.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            lr=args.lr,
            num_of_epochs=args.num_of_epochs,
            train_steps=len(train_files),
            val_steps=len(val_files),
            loss_heatmap=args.loss_heatmap,
            loss_quality=args.loss_quality,
            train_mode=args.train_mode,
            c_bg=args.c_bg,
            output_path=args.output,
            best_train_loss_output=args.best_train_loss_output,
            log_file=args.log_file
        )

if __name__ == '__main__':
    main()
