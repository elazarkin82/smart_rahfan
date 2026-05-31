import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import os
import pickle
import numpy as np

# =====================================================================
# Custom Losses for Continuous Heatmap Regression
# =====================================================================

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

# =====================================================================
# Target Tracker Ver 4 Class
# =====================================================================

class TargetTrackerVer4:
    def __init__(self, ref_shape=(16, 16, 16, 3), search_shape=(256, 256, 3)):
        self.ref_shape = ref_shape
        self.search_shape = search_shape
        self.model = None

    def _inverted_residual_block(self, inputs, expansion, filters, strides, name_prefix):
        x = inputs
        in_channels = inputs.shape[-1]
        
        # Expand
        if expansion > 1:
            x = layers.Conv2D(expansion * in_channels, (1, 1), padding="same", use_bias=False, name=f"{name_prefix}_expand")(x)
            x = layers.BatchNormalization(name=f"{name_prefix}_expand_bn")(x)
            x = layers.ReLU(6.0, name=f"{name_prefix}_expand_relu")(x)
            
        # Depthwise
        x = layers.DepthwiseConv2D((3, 3), strides=strides, padding="same", use_bias=False, name=f"{name_prefix}_dw")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
        x = layers.ReLU(6.0, name=f"{name_prefix}_dw_relu")(x)
        
        # Project
        x = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False, name=f"{name_prefix}_project")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_project_bn")(x)
        
        if strides == 1 and in_channels == filters:
            x = layers.Add(name=f"{name_prefix}_add")([inputs, x])
            
        return x

    def _create_search_backbone(self):
        inputs = layers.Input(shape=self.search_shape, name="search_backbone_input")
        
        # Init conv (strides=2) -> (128, 128, 16)
        x = layers.Conv2D(16, (3, 3), strides=2, padding="same", use_bias=False, name="sb_init_conv")(inputs)
        x = layers.BatchNormalization(name="sb_init_bn")(x)
        x = layers.ReLU(6.0, name="sb_init_relu")(x)
        
        # IR Block 1 (strides=2) -> (64, 64, 24)
        x = self._inverted_residual_block(x, expansion=2, filters=24, strides=2, name_prefix="sb_ir1")
        
        # IR Block 2 (strides=2) -> (32, 32, 32)
        x = self._inverted_residual_block(x, expansion=3, filters=32, strides=2, name_prefix="sb_ir2")
        
        # IR Block 3 (strides=2) -> (16, 16, 64)
        x = self._inverted_residual_block(x, expansion=4, filters=64, strides=2, name_prefix="sb_ir3")
        
        # Final Expand -> (16, 16, 128)
        x = layers.Conv2D(128, (1, 1), padding="same", use_bias=False, name="sb_final_conv")(x)
        x = layers.BatchNormalization(name="sb_final_bn")(x)
        x = layers.ReLU(6.0, name="sb_final_relu")(x)
        
        return models.Model(inputs, x, name="search_feature_extractor")

    def _create_reference_encoder(self):
        inputs = layers.Input(shape=self.ref_shape, name="ref_encoder_input")
        
        # Reshape (16, 16, 16, 3) -> (16, 16, 48)
        x = layers.Reshape((self.ref_shape[0], self.ref_shape[1], self.ref_shape[2] * self.ref_shape[3]), name="ref_reshape")(inputs)
        
        # Conv 1
        x = layers.Conv2D(32, (3, 3), strides=2, padding="same", use_bias=False, name="ref_init_conv")(x) # -> (8, 8, 32)
        x = layers.BatchNormalization(name="ref_init_bn")(x)
        x = layers.ReLU(6.0, name="ref_init_relu")(x)
        
        # IR Block
        x = self._inverted_residual_block(x, expansion=2, filters=64, strides=2, name_prefix="ref_ir1") # -> (4, 4, 64)
        
        # Final Expand to match search feature channels
        x = layers.Conv2D(128, (1, 1), padding="same", use_bias=False, name="ref_final_conv")(x)
        x = layers.BatchNormalization(name="ref_final_bn")(x)
        x = layers.ReLU(6.0, name="ref_final_relu")(x)
        
        # Global Average Pooling to get the vector
        x = layers.GlobalAveragePooling2D(name="ref_gap")(x) # -> (128,)
        
        # Expand dims to allow broadcast multiplication (1, 1, 128)
        x = layers.Reshape((1, 1, 128), name="ref_vector_reshape")(x)
        
        return models.Model(inputs, x, name="reference_target_encoder")

    def create_model(self):
        ref_input = layers.Input(shape=self.ref_shape, name="reference_stack")
        search_input = layers.Input(shape=self.search_shape, name="search_frame")
        
        # 1. Encoders
        ref_encoder = self._create_reference_encoder()
        search_encoder = self._create_search_backbone()
        
        target_vector = ref_encoder(ref_input)          # (1, 1, 128)
        search_features = search_encoder(search_input)  # (16, 16, 128)
        
        # 2. Late Feature Gating (Channel-wise Attention)
        # Multiply search features by the target representation
        fused_features = layers.Multiply(name="late_feature_gating")([search_features, target_vector])
        
        # 3. Decoder
        # Upsample 1: (16, 16) -> (32, 32)
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_up1")(fused_features)
        x = self._inverted_residual_block(x, expansion=2, filters=64, strides=1, name_prefix="decoder_ir1")
        
        # Upsample 2: (32, 32) -> (64, 64)
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_up2")(x)
        x = self._inverted_residual_block(x, expansion=2, filters=32, strides=1, name_prefix="decoder_ir2")
        
        # Upsample 3: (64, 64) -> (128, 128)
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_up3")(x)
        x = self._inverted_residual_block(x, expansion=2, filters=16, strides=1, name_prefix="decoder_ir3")
        
        # Upsample 4: (128, 128) -> (256, 256)
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_up4")(x)
        x = self._inverted_residual_block(x, expansion=2, filters=8, strides=1, name_prefix="decoder_ir4")
        
        # Final prediction layer
        output_heatmap = layers.Conv2D(1, (3, 3), padding="same", activation="sigmoid", name="predicted_heatmap")(x)
        
        self.model = models.Model(
            inputs=[ref_input, search_input],
            outputs=output_heatmap,
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

    def evaluate(self, dataset, loss_fn):
        val_loss_avg = tf.keras.metrics.Mean()
        import tqdm
        for inputs, targets in tqdm.tqdm(dataset, desc="Evaluating", leave=False):
            predictions = self.model(inputs, training=False)
            loss_value = loss_fn(targets, predictions)
            val_loss_avg.update_state(loss_value)
        return float(val_loss_avg.result())

    def train_epoch(self, dataset, optimizer, loss_fn, epoch, num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        import tqdm
        progress_bar = tqdm.tqdm(dataset, desc=f"Epoch {epoch:03d}/{num_epochs:03d}", leave=False)
        
        for inputs, targets in progress_bar:
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                loss_value = loss_fn(targets, predictions)
                
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            epoch_loss_avg.update_state(loss_value)
            progress_bar.set_postfix(loss=float(loss_value))
            
        return float(epoch_loss_avg.result())

    def train(self, train_dataset, val_dataset, lr, num_of_epochs, loss_name="mse", output_path=None, best_train_loss_output=None, log_file=None):
        if loss_name == "mse":
            loss_fn = losses.MeanSquaredError()
        elif loss_name == "dice_bce":
            loss_fn = dice_bce_loss
        elif loss_name == "focal":
            loss_fn = focal_loss
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
            
        optimizer = optimizers.Adam(learning_rate=lr)
        
        self.log("Calculating initial validation loss...", log_file)
        best_val_loss = self.evaluate(val_dataset, loss_fn)
        best_train_loss = float('inf')
        self.log(f"Initial Validation Loss: {best_val_loss:.6f}", log_file)
        
        for epoch in range(1, num_of_epochs + 1):
            epoch_loss = self.train_epoch(train_dataset, optimizer, loss_fn, epoch, num_of_epochs)
            
            if best_train_loss_output and epoch_loss < best_train_loss:
                self.log(f"   [TRAIN IMPROVEMENT] Train loss improved from {best_train_loss:.6f} to {epoch_loss:.6f}. Saving to {best_train_loss_output}", log_file)
                best_train_loss = epoch_loss
                if os.path.dirname(best_train_loss_output):
                    os.makedirs(os.path.dirname(best_train_loss_output), exist_ok=True)
                self.model.save(best_train_loss_output)
            
            val_loss = self.evaluate(val_dataset, loss_fn)
            self.log(f"Epoch {epoch:03d}/{num_of_epochs:03d} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}", log_file)
            
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
    
    # We yield individual examples from the PKL
    for sample in samples:
        ref = sample['reference_stack']
        search = sample['search_frame']
        heatmap = sample['ground_truth_heatmap']
        
        yield (ref, search), heatmap

def build_tf_dataset(pkl_files, batch_size=16, shuffle=True):
    if not pkl_files:
        raise ValueError("No PKL files provided for dataset.")
        
    def generator():
        local_files = list(pkl_files)
        if shuffle:
            import random
            random.shuffle(local_files)
        for path in local_files:
            yield from parse_training_samples(path)
            
    # Reference shape: (16, 16, 16, 3) uint8
    # Search frame shape: (H, W, 3) uint8
    # Heatmap shape: (H, W, 1) float16
    output_signature = (
        (
            tf.TensorSpec(shape=(16, 16, 16, 3), dtype=tf.uint8, name="reference_stack"),
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name="search_frame")
        ),
        tf.TensorSpec(shape=(None, None, 1), dtype=tf.float16, name="heatmap")
    )
    
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    
    def process_element(inputs, heatmap):
        ref, search = inputs
        
        # Resize inputs to model expectations
        # Reference is already 16x16x16x3, just cast and normalize
        ref_float = tf.cast(ref, tf.float32) / 255.0
        
        # Resize search frame to 256x256
        search_resized = tf.image.resize(search, (256, 256))
        search_float = tf.cast(search_resized, tf.float32) / 255.0
        
        # Resize heatmap to 256x256
        heatmap_resized = tf.image.resize(tf.cast(heatmap, tf.float32), (256, 256))
        
        return {"reference_stack": ref_float, "search_frame": search_float}, heatmap_resized

    ds = ds.map(process_element, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(100)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return ds

def main():
    import argparse
    parser = argparse.ArgumentParser(description="TargetTrackerVer4 Training CLI")
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--dataset_dir", required=True, help="Path to dataset PKL dir")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval_pkl_num", type=int, default=4, help="Number of PKLs for validation")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_of_epochs", type=int, default=10)
    parser.add_argument("--loss", choices=["mse", "dice_bce", "focal"], default="mse")
    parser.add_argument("--output", type=str, default="outputs/tracker.keras")
    parser.add_argument("--best_train_loss_output", type=str, default="outputs/tracker_best_train.keras")
    parser.add_argument("--init_keras_file", type=str, default=None, help="Path to initial model to resume from")
    parser.add_argument("--log_file", type=str, default="outputs/train.log")
    
    args = parser.parse_args()
    
    if args.command == "train":
        import glob
        all_pkls = sorted(glob.glob(os.path.join(args.dataset_dir, "train_*.pkl")))
        if not all_pkls:
            raise FileNotFoundError(f"No train_*.pkl files in {args.dataset_dir}")
            
        val_files = all_pkls[:args.eval_pkl_num]
        train_files = all_pkls[args.eval_pkl_num:]
        if not train_files:
            train_files = val_files # Fallback
            
        train_ds = build_tf_dataset(train_files, batch_size=args.batch_size, shuffle=True)
        val_ds = build_tf_dataset(val_files, batch_size=args.batch_size, shuffle=False)
        
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
            loss_name=args.loss,
            output_path=args.output,
            best_train_loss_output=args.best_train_loss_output,
            log_file=args.log_file
        )

if __name__ == '__main__':
    main()
