import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import os
import pickle
import numpy as np

# =====================================================================
# Custom Losses for Continuous Heatmap Regression
# =====================================================================

@tf.keras.utils.register_keras_serializable(package="Custom")
def dbsz_hard_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    mask_high = tf.cast(y_true >= 0.5, tf.float32)
    mask_low = tf.cast(y_true <= 0.01, tf.float32)
    
    n_high = tf.reduce_sum(mask_high) + 1e-5
    n_low = tf.reduce_sum(mask_low) + 1e-5
    
    loss_high = tf.reduce_sum(mask_high * tf.abs(y_true - y_pred)) / n_high
    loss_low = tf.reduce_sum(mask_low * tf.square(y_true - y_pred)) / n_low
    
    return loss_high + loss_low

@tf.keras.utils.register_keras_serializable(package="Custom")
def dbsz_soft_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    w_high = tf.pow(y_true, 4.0)
    w_low = tf.pow(1.0 - y_true, 16.0)
    
    loss_high = tf.reduce_sum(w_high * tf.abs(y_true - y_pred)) / (tf.reduce_sum(w_high) + 1e-5)
    loss_low = tf.reduce_sum(w_low * tf.square(y_true - y_pred)) / (tf.reduce_sum(w_low) + 1e-5)
    
    return loss_high + loss_low

@tf.keras.utils.register_keras_serializable(package="Custom")
def dbsz_relu_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    w_high = tf.nn.relu(y_true - 0.5)
    w_low = tf.nn.relu(0.2 - y_true)
    
    k1 = 1.0 / (tf.reduce_sum(w_high) + 1e-5)
    k = 1.0 / (tf.reduce_sum(w_low) + 1e-5)
    
    loss_high = k1 * tf.reduce_sum(w_high * tf.square(y_true - y_pred))
    loss_low = k * tf.reduce_sum(w_low * tf.square(y_true - y_pred))
    
    return loss_high + loss_low

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
    
    # 1. Continuous Focal Loss
    bce_pos = - y_true * tf.math.log(y_pred) * tf.math.pow(1.0 - y_pred, gamma)
    bce_neg = - (1.0 - y_true) * tf.math.log(1.0 - y_pred) * tf.math.pow(y_pred, gamma)
    focal_loss_val = tf.reduce_mean(alpha * bce_pos + (1.0 - alpha) * bce_neg)
    
    # 2. Soft Dice Loss (Square Form) for continuous heatmaps
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
    
    # Soft continuous formulation of CenterNet Penalty-Reduced Focal Loss:
    # y_true acts as the anchor, and (1 - y_true)^beta suppresses background penalty near the peak
    pos_loss = - y_true * tf.math.pow(1.0 - y_pred, alpha) * tf.math.log(y_pred)
    neg_loss = - tf.math.pow(1.0 - y_true, beta) * tf.math.pow(y_pred, alpha) * tf.math.log(1.0 - y_pred)
    
    loss = pos_loss + neg_loss
    return tf.reduce_mean(loss)

@tf.keras.utils.register_keras_serializable(package="Custom")
def centernet_dice_loss(y_true, y_pred, focal_weight=1.0, dice_weight=1.0, alpha=2.0, beta=4.0):
    eps = 1e-7
    # 1. CenterNet Focal Loss
    cn_loss = centernet_focal_loss(y_true, y_pred, alpha=alpha, beta=beta)
    
    # 2. Soft Dice Loss (Square Form)
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(tf.square(y_true_f)) + tf.reduce_sum(tf.square(y_pred_f))
    dice_loss_val = 1.0 - (2.0 * intersection + eps) / (denominator + eps)
    
    return focal_weight * cn_loss + dice_weight * dice_loss_val

# =====================================================================
# Target Tracker Ver 4 Class
# =====================================================================

class TargetTrackerVer4:
    def __init__(self, ref_shape=(16, 32, 32, 1), search_shape=(256, 256, 1)):
        self.ref_shape = ref_shape
        self.search_shape = search_shape
        self.model = None

    def _inverted_residual_block(self, inputs, expansion, filters, strides, name_prefix):
        x = inputs
        in_channels = inputs.shape[-1]
        
        # Expand
        if expansion > 1:
            x = layers.Conv2D(expansion * in_channels, (1, 1), padding="same", use_bias=False, name=f"{name_prefix}_expand")(x)
            x = layers.GroupNormalization(groups=8, name=f"{name_prefix}_expand_gn")(x)
            x = layers.ReLU(6.0, name=f"{name_prefix}_expand_relu")(x)
            
        # Depthwise
        x = layers.DepthwiseConv2D((3, 3), strides=strides, padding="same", use_bias=False, name=f"{name_prefix}_dw")(x)
        x = layers.GroupNormalization(groups=8, name=f"{name_prefix}_dw_gn")(x)
        x = layers.ReLU(6.0, name=f"{name_prefix}_dw_relu")(x)
        
        # Project
        x = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False, name=f"{name_prefix}_project")(x)
        x = layers.GroupNormalization(groups=8, name=f"{name_prefix}_project_gn")(x)
        
        if strides == 1 and in_channels == filters:
            x = layers.Add(name=f"{name_prefix}_add")([inputs, x])
            
        return x

    def _create_search_backbone(self):
        inputs = layers.Input(shape=self.search_shape, name="search_backbone_input")
        
        # Init conv (strides=2) -> (128, 128, 16)
        x1 = layers.Conv2D(16, (3, 3), strides=2, padding="same", use_bias=False, name="sb_init_conv")(inputs)
        x1 = layers.GroupNormalization(groups=8, name="sb_init_gn")(x1)
        x1 = layers.ReLU(6.0, name="sb_init_relu")(x1)
        
        # IR Block 1 (strides=2) -> (64, 64, 24)
        x2 = self._inverted_residual_block(x1, expansion=2, filters=24, strides=2, name_prefix="sb_ir1")
        
        # IR Block 2 (strides=2) -> (32, 32, 32)
        x3 = self._inverted_residual_block(x2, expansion=3, filters=32, strides=2, name_prefix="sb_ir2")
        
        # IR Block 3 (strides=2) -> (16, 16, 64)
        x4 = self._inverted_residual_block(x3, expansion=4, filters=64, strides=2, name_prefix="sb_ir3")
        
        # Final Expand -> (16, 16, 128)
        x = layers.Conv2D(128, (1, 1), padding="same", use_bias=False, name="sb_final_conv")(x4)
        x = layers.GroupNormalization(groups=8, name="sb_final_gn")(x)
        x = layers.ReLU(6.0, name="sb_final_relu")(x)
        
        return models.Model(inputs, [x1, x2, x3, x], name="search_feature_extractor")

    def _create_reference_encoder(self):
        inputs = layers.Input(shape=self.ref_shape, name="ref_encoder_input")
        
        # 1. Permute to align dimensions properly: (Layers, H, W, C) -> (H, W, Layers, C)
        x = layers.Permute((2, 3, 1, 4), name="ref_permute")(inputs)
        
        # 2. Reshape to combine Layers and Channels: (32, 32, 16, 1) -> (32, 32, 16)
        x = layers.Reshape((self.ref_shape[1], self.ref_shape[2], self.ref_shape[0] * self.ref_shape[3]), name="ref_reshape")(x)
        
        # Conv 1
        x = layers.Conv2D(32, (3, 3), strides=2, padding="same", use_bias=False, name="ref_init_conv")(x) # -> (16, 16, 32)
        x = layers.GroupNormalization(groups=8, name="ref_init_gn")(x)
        x = layers.ReLU(6.0, name="ref_init_relu")(x)
        
        # IR Block
        x = self._inverted_residual_block(x, expansion=2, filters=64, strides=2, name_prefix="ref_ir1") # -> (8, 8, 64)
        
        # Final Expand to match search feature channels
        x = layers.Conv2D(128, (1, 1), padding="same", use_bias=False, name="ref_final_conv")(x)
        x = layers.GroupNormalization(groups=8, name="ref_final_gn")(x)
        x = layers.ReLU(6.0, name="ref_final_relu")(x)
        
        # Return spatial features of shape (8, 8, 128) for spatial cross-attention
        return models.Model(inputs, x, name="reference_target_encoder")

    def create_model(self):
        ref_input = layers.Input(shape=self.ref_shape, name="reference_stack")
        search_input = layers.Input(shape=self.search_shape, name="search_frame")
        
        # 1. Encoders
        ref_encoder = self._create_reference_encoder()
        search_encoder = self._create_search_backbone()
        
        ref_features = ref_encoder(ref_input)          # (8, 8, 128)
        
        # Multi-output Search Encoder to get intermediate features for skip connections
        sb_init, sb_ir1, sb_ir2, search_features = search_encoder(search_input)
        
        # 2. Dot-Product Cross-Attention Fusion
        # Project Q from search_features, K and V from ref_features
        q_proj = layers.Conv2D(64, (1, 1), use_bias=False, name="q_proj")(search_features)  # (16, 16, 64)
        k_proj = layers.Conv2D(64, (1, 1), use_bias=False, name="k_proj")(ref_features)     # (8, 8, 64)
        v_proj = layers.Conv2D(128, (1, 1), use_bias=False, name="v_proj")(ref_features)    # (8, 8, 128)
        
        # Flatten spatial dimensions
        q_flat = layers.Reshape((256, 64), name="q_flat")(q_proj)
        k_flat = layers.Reshape((64, 64), name="k_flat")(k_proj)
        v_flat = layers.Reshape((64, 128), name="v_flat")(v_proj)
        
        # Calculate search-to-reference spatial correlation: (256, 64) x (64, 64)^T -> (256, 64)
        attn_weights = layers.Dot(axes=(2, 2), name="attention_dot")([q_flat, k_flat])
        attn_weights = layers.Lambda(lambda x: x / 8.0, name="attention_scale")(attn_weights) # scale by sqrt(d_k)=8
        attn_weights = layers.Softmax(axis=-1, name="attention_softmax")(attn_weights)
        
        # Fused features: (256, 64) x (64, 128) -> (256, 128)
        fused_flat = layers.Dot(axes=(2, 1), name="attention_value_dot")([attn_weights, v_flat])
        fused_features = layers.Reshape((16, 16, 128), name="fused_features_reshape")(fused_flat)
        
        # 3. Decoder for Output 1: Heatmap (with U-Net style skip connections)
        # Upsample 1: (16, 16) -> (32, 32)
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_up1")(fused_features)
        # Skip 1: Concatenate with sb_ir2 (32, 32, 32)
        x = layers.Concatenate(axis=-1, name="decoder_skip1")([x, sb_ir2])
        x = self._inverted_residual_block(x, expansion=2, filters=64, strides=1, name_prefix="decoder_ir1")
        
        # Upsample 2: (32, 32) -> (64, 64)
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_up2")(x)
        # Skip 2: Concatenate with sb_ir1 (64, 64, 24)
        x = layers.Concatenate(axis=-1, name="decoder_skip2")([x, sb_ir1])
        x = self._inverted_residual_block(x, expansion=2, filters=32, strides=1, name_prefix="decoder_ir2")
        
        # Upsample 3: (64, 64) -> (128, 128)
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_up3")(x)
        # Skip 3: Concatenate with sb_init (128, 128, 16)
        x = layers.Concatenate(axis=-1, name="decoder_skip3")([x, sb_init])
        x = self._inverted_residual_block(x, expansion=2, filters=16, strides=1, name_prefix="decoder_ir3")
        
        # Upsample 4: (128, 128) -> (256, 256)
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_up4")(x)
        x = self._inverted_residual_block(x, expansion=2, filters=8, strides=1, name_prefix="decoder_ir4")
        
        # Final prediction heatmap
        output_heatmap = layers.Conv2D(1, (3, 3), padding="same", activation="sigmoid", name="predicted_heatmap")(x)
        
        # 4. Classification Branch for Output 2: Quality Score
        q = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu", name="quality_conv")(fused_features) # -> (8, 8, 32)
        q = layers.GlobalAveragePooling2D(name="quality_gap")(q) # -> (32,)
        q = layers.Dense(16, activation="relu", name="quality_fc1")(q)
        output_quality = layers.Dense(1, activation="sigmoid", name="predicted_quality")(q)
        
        self.model = models.Model(
            inputs=[ref_input, search_input],
            outputs=[output_heatmap, output_quality],
            name="TargetTrackerVer4"
        )
        
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

    def evaluate(self, dataset, loss_fn_heatmap, loss_fn_quality, steps=None):
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
            
            # Mask out negative samples from the heatmap loss calculation
            is_pos = tf.reshape(gt_quality, [-1]) > 0.0
            any_pos = tf.reduce_any(is_pos)
            
            if any_pos:
                pos_gt_heatmap = tf.boolean_mask(gt_heatmap, is_pos)
                pos_pred_heatmap = tf.boolean_mask(pred_heatmap, is_pos)
                loss_heatmap = loss_fn_heatmap(pos_gt_heatmap, pos_pred_heatmap)
            else:
                loss_heatmap = tf.constant(0.0, dtype=tf.float32)
                
            loss_quality = loss_fn_quality(gt_quality, pred_quality)
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

    def train_epoch(self, dataset, optimizer, loss_fn_heatmap, loss_fn_quality, epoch, num_epochs, steps=None):
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
                
                # Mask out negative samples from the heatmap loss calculation
                is_pos = tf.reshape(gt_quality, [-1]) > 0.0
                any_pos = tf.reduce_any(is_pos)
                
                if any_pos:
                    pos_gt_heatmap = tf.boolean_mask(gt_heatmap, is_pos)
                    pos_pred_heatmap = tf.boolean_mask(pred_heatmap, is_pos)
                    loss_heatmap = loss_fn_heatmap(pos_gt_heatmap, pos_pred_heatmap)
                else:
                    loss_heatmap = tf.constant(0.0, dtype=tf.float32)
                    
                loss_quality = loss_fn_quality(gt_quality, pred_quality)
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

    def train(self, train_dataset, val_dataset, lr, num_of_epochs, train_steps=None, val_steps=None, loss_heatmap="adaptive_wing", loss_quality="bce", output_path=None, best_train_loss_output=None, log_file=None):
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
            loss_fn_heatmap = dbsz_hard_loss
        elif loss_heatmap == "dbsz_soft":
            loss_fn_heatmap = dbsz_soft_loss
        elif loss_heatmap == "dbsz_relu":
            loss_fn_heatmap = dbsz_relu_loss
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
        optimizer = optimizers.Adam(learning_rate=lr, jit_compile=False)
        
        self.log("Calculating initial validation loss...", log_file)
        best_val_loss, init_val_hm, init_val_q = self.evaluate(val_dataset, loss_fn_heatmap, loss_fn_quality, steps=val_steps)
        best_train_loss = float('inf')
        self.log(f"Initial Validation Loss: {best_val_loss:.6f} (HM: {init_val_hm:.6f}, Q: {init_val_q:.6f})", log_file)
        
        for epoch in range(1, num_of_epochs + 1):
            epoch_loss, epoch_hm, epoch_q = self.train_epoch(train_dataset, optimizer, loss_fn_heatmap, loss_fn_quality, epoch, num_of_epochs, steps=train_steps)
            
            if best_train_loss_output and epoch_loss < best_train_loss:
                self.log(f"   [TRAIN IMPROVEMENT] Train loss improved from {best_train_loss:.6f} to {epoch_loss:.6f}. Saving to {best_train_loss_output}", log_file)
                best_train_loss = epoch_loss
                if os.path.dirname(best_train_loss_output):
                    os.makedirs(os.path.dirname(best_train_loss_output), exist_ok=True)
                self.model.save(best_train_loss_output)
            
            val_loss, val_hm, val_q = self.evaluate(val_dataset, loss_fn_heatmap, loss_fn_quality, steps=val_steps)
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
    
    # Stack list of dicts to form raw batched numpy arrays
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
            
    # Batch output signature: added leading Batch dimension (None) to support dynamic batch size
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
        
        # Resize inputs to model expectations (tf.image.resize and cast natively support batches)
        ref_float = tf.cast(ref, tf.float32) / 255.0
        
        # Resize search frame to 256x256
        search_resized = tf.image.resize(search, (256, 256))
        search_float = tf.cast(search_resized, tf.float32) / 255.0
        
        # Resize heatmap to 256x256
        heatmap_resized = tf.image.resize(tf.cast(heatmap, tf.float32), (256, 256))
        
        return {"reference_stack": ref_float, "search_frame": search_float}, {"predicted_heatmap": heatmap_resized, "predicted_quality": tf.cast(quality, tf.float32)}

    ds = ds.map(process_element, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

def main():
    import argparse
    parser = argparse.ArgumentParser(description="TargetTrackerVer4 Training CLI")
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--dataset_dir", required=True, help="Path to dataset PKL dir")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size (ignored, defined by dataset files)")
    parser.add_argument("--eval_pkl_num", type=int, default=4, help="Number of PKLs for validation")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_of_epochs", type=int, default=10)
    parser.add_argument("--loss_heatmap", choices=["mse", "dice_bce", "focal", "focal_dice", "centernet", "centernet_dice", "adaptive_wing", "dbsz_hard", "dbsz_soft", "dbsz_relu"], default="dbsz_soft")
    parser.add_argument("--loss_quality", choices=["bce", "mse", "huber", "logcosh"], default="bce")
    parser.add_argument("--output", type=str, default="outputs/tracker.keras")
    parser.add_argument("--best_train_loss_output", type=str, default="outputs/tracker_best_train.keras")
    parser.add_argument("--init_keras_file", type=str, default=None, help="Path to initial model to resume from")
    parser.add_argument("--log_file", type=str, default="outputs/train.log")
    
    args = parser.parse_args()
    
    if args.command == "train":
        import glob
        all_pkls = sorted(glob.glob(os.path.join(args.dataset_dir, "batch_*.pkl")))
        if not all_pkls:
            raise FileNotFoundError(f"No batch_*.pkl files in {args.dataset_dir}")
            
        val_files = all_pkls[:args.eval_pkl_num]
        train_files = all_pkls[args.eval_pkl_num:]
        if not train_files:
            train_files = val_files # Fallback
            
        train_ds = build_tf_dataset(train_files, shuffle=True)
        val_ds = build_tf_dataset(val_files, shuffle=False)
        
        tracker = TargetTrackerVer4()
        
        if args.init_keras_file and os.path.exists(args.init_keras_file):
            import tensorflow as tf
            print(f"Resuming training: loading model from {args.init_keras_file}...")
            tracker.model = tf.keras.models.load_model(args.init_keras_file, compile=False, safe_mode=False)
        else:
            print("Building new TargetTrackerVer4 model...")
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
            output_path=args.output,
            best_train_loss_output=args.best_train_loss_output,
            log_file=args.log_file
        )

if __name__ == '__main__':
    main()
