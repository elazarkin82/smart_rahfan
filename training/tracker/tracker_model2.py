import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import os
import cv2
import pickle
import numpy as np

# =====================================================================
# Helper Functions for Circular Attention Masking and Gaussian Heatmaps
# =====================================================================

def apply_circle_mask(img_uint8, coords, radius):
    """
    Applies a binary circular mask to a grayscale image (0-255 uint8)
    centered around the normalized coordinate, blacking out the region outside.
    
    Args:
        img_uint8 (np.ndarray): Grayscale image array of shape (H, W).
        coords (list/tuple): Normalized [x, y] coordinates in [0, 1] range.
        radius (int): Radius of the circular mask in pixels.
        
    Returns:
        np.ndarray: Grayscale image with the mask applied.
    """
    h, w = img_uint8.shape[:2]
    x_px = int(coords[0] * w)
    y_px = int(coords[1] * h)
    
    # Create binary circular mask
    mask = np.zeros_like(img_uint8)
    cv2.circle(mask, (x_px, y_px), radius, 255, -1)
    
    # Keep only the region inside the circle
    masked_img = cv2.bitwise_and(img_uint8, mask)
    return masked_img

def generate_gaussian_heatmap(coords, size=64, sigma=4.0):
    """
    Generates a 2D Gaussian heatmap centered around the normalized coordinate
    to provide a continuous target peak for smooth spatial gradients.
    
    Args:
        coords (list/tuple): Normalized [x, y] coordinate in [0, 1] range.
        size (int): Dimensions of the output heatmap grid (default: 64).
        sigma (float): Standard deviation of the Gaussian kernel (default: 4.0).
        
    Returns:
        np.ndarray: Gaussian heatmap array of shape (size, size, 1).
    """
    x_target = coords[0] * size
    y_target = coords[1] * size
    
    x = np.arange(0, size, 1, dtype=np.float32)
    y = np.arange(0, size, 1, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Compute squared Euclidean distance from the center target point
    d2 = (x_grid - x_target) ** 2 + (y_grid - y_target) ** 2
    # Apply standard 2D Gaussian formula
    heatmap = np.exp(-d2 / (2.0 * sigma ** 2))
    
    # Expand channel dimension
    return np.expand_dims(heatmap, axis=-1)

# =====================================================================
# Custom Losses for Spatial Heatmap Regression / Segmentation
# =====================================================================

@tf.keras.utils.register_keras_serializable(package="Custom")
def dice_bce_loss(y_true, y_pred, bce_weight=1.0, dice_weight=1.0):
    """
    Joint Binary Cross-Entropy (BCE) and Dice Loss.
    Combines pixel-wise probability constraints with overall structural overlap.
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    # 1. Binary Cross Entropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce_loss = tf.reduce_mean(bce)
    
    # 2. Dice Loss
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    eps = 1e-12
    dice_coef = (2. * intersection + eps) / (union + eps)
    dice_loss = 1.0 - dice_coef
    
    return bce_weight * bce_loss + dice_weight * dice_loss

@tf.keras.utils.register_keras_serializable(package="Custom")
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    100% Stable and NaN-proof Sigmoid Focal Loss for continuous/Gaussian heatmaps.
    Uses safe clipping and avoids division-by-zero gradients.
    """
    # Clip predictions to prevent log(0) and division by zero gradients
    eps = 1e-12
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    
    # Calculate binary cross entropy terms safely
    bce_pos = - y_true * tf.math.log(y_pred) * tf.math.pow(1.0 - y_pred, gamma)
    bce_neg = - (1.0 - y_true) * tf.math.log(1.0 - y_pred) * tf.math.pow(y_pred, gamma)
    
    # Combine and scale
    loss = alpha * bce_pos + (1.0 - alpha) * bce_neg
    return tf.reduce_mean(loss)

# =====================================================================
# Fully Convolutional Target Tracker Class
# =====================================================================

class TargetTracker2:
    """
    A Fully Convolutional recursive target tracker in Keras.
    
    To support edge deployment on NPUs (like RK3566) and prevent loss of spatial
    resolution, this model:
      - Removes continuous coordinate inputs and Dense bottleneck layers.
      - Receives visual attention masks baked directly into the input frames:
        1. hist_frame_masked: (256, 256, 1) masked with a large radius (128px)
        2. prev_frame_masked: (256, 256, 1) masked with a smaller search radius (50px)
        3. curr_frame: (256, 256, 1) unmasked search region
      - Regresses a spatial probability distribution (Gaussian Heatmap) of shape (64, 64, 1).
      - Sub-pixel tracking coordinates can be extracted via CPU-based Center of Mass post-processing.
    """
    def __init__(self, input_shape=(256, 256, 1)):
        self.input_shape = input_shape
        self.model = None
        
    def _create_cnn_backbone(self):
        """
        Creates a shared Siamese CNN backbone for feature extraction.
        Output shape: (16, 16, 256)
        """
        inputs = layers.Input(shape=self.input_shape, name="cnn_input")
        
        # Stride-2 convolutions progressively downsample while doubling depth
        x = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(64, (3, 3), strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(128, (3, 3), strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(256, (3, 3), strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        
        return models.Model(inputs, x, name="siamese_cnn_backbone")

    def create_model(self):
        """
        Builds the Fully Convolutional Network (FCN) functional model.
        Returns:
            tf.keras.Model: The uncompiled Keras FCN model.
        """
        # Inputs: 3 frames (2 masked, 1 unmasked)
        hist_frame_in = layers.Input(shape=self.input_shape, name="hist_frame")
        prev_frame_in = layers.Input(shape=self.input_shape, name="prev_frame")
        curr_frame_in = layers.Input(shape=self.input_shape, name="curr_frame")
        
        # Shared Siamese Backbone
        cnn_backbone = self._create_cnn_backbone()
        
        # Extract spatial features at (16, 16, 256)
        hist_features = cnn_backbone(hist_frame_in)
        prev_features = cnn_backbone(prev_frame_in)
        curr_features = cnn_backbone(curr_frame_in)
        
        # Motion Dynamics (spatial feature subtraction)
        motion_hist_to_prev = layers.Subtract(name="visual_motion_hist_to_prev")([prev_features, hist_features])
        motion_prev_to_curr = layers.Subtract(name="visual_motion_prev_to_curr")([curr_features, prev_features])
        
        # Concatenate features along the channel axis (16, 16, 1280)
        fused_features = layers.Concatenate(axis=-1, name="temporal_visual_fusion")(
            [hist_features, prev_features, curr_features, motion_hist_to_prev, motion_prev_to_curr]
        )
        
        # Process fused representations
        x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="displacement_conv")(fused_features)
        x = layers.BatchNormalization()(x)
        
        # Decoder Stage 1: (16, 16) -> (32, 32)
        x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding="same", activation="relu", name="decoder_upsample_1")(x)
        x = layers.BatchNormalization()(x)
        
        # Decoder Stage 2: (32, 32) -> (64, 64)
        x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu", name="decoder_upsample_2")(x)
        x = layers.BatchNormalization()(x)
        
        # Final Convolutional Prediction Layer: (64, 64, 1) probability heatmap
        output_heatmap = layers.Conv2D(1, (3, 3), padding="same", activation="sigmoid", name="predicted_heatmap")(x)
        
        self.model = models.Model(
            inputs=[hist_frame_in, prev_frame_in, curr_frame_in],
            outputs=output_heatmap,
            name="TargetTrackerFCN"
        )
        
        return self.model

    @staticmethod
    def generate_dataset(images_path, output_path, batch_size=256, num_of_samples=16384):
        """
        Generates synthetic masked tracking dataset from images, saving debug coords.
        """
        print("Starting masked dataset generation crawling...")
        
        image_extensions = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")
        image_paths = []
        
        if os.path.isdir(images_path):
            for root, _, files in os.walk(images_path):
                for f in files:
                    if f.lower().endswith(image_extensions):
                        image_paths.append(os.path.join(root, f))
            if not image_paths:
                raise ValueError(f"No images found in directory {images_path}")
        elif os.path.isfile(images_path):
            with open(images_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                path = line.strip()
                if path and path.lower().endswith(image_extensions):
                    if os.path.exists(path):
                        image_paths.append(path)
            if not image_paths:
                raise ValueError(f"No valid image paths listed in {images_path}")
        else:
            raise ValueError(f"Invalid images_path: {images_path}")
            
        print(f"Found {len(image_paths)} valid images.")
        
        num_batches = int(np.ceil(num_of_samples / batch_size))
        total_samples = num_batches * batch_size
        print(f"Generating {total_samples} samples ({num_batches} batches of size {batch_size}).")
        
        os.makedirs(output_path, exist_ok=True)
        
        def apply_parallax_warp(img_256, coords):
            angle = np.random.uniform(-20.0, 20.0)
            R = cv2.getRotationMatrix2D((128.0, 128.0), angle, 1.0)
            rotated_img = cv2.warpAffine(img_256, R, (256, 256), borderMode=cv2.BORDER_REPLICATE)
            
            x_px, y_px = coords[0] * 256.0, coords[1] * 256.0
            pt_rot = np.array([x_px, y_px, 1.0], dtype=np.float32)
            rotated_pt = np.dot(R, pt_rot)
            
            src_pts = np.float32([[15, 15], [240, 15], [15, 240]])
            delta = 25.0
            dst_pts = src_pts + np.random.uniform(-delta, delta, src_pts.shape).astype(np.float32)
            
            M = cv2.getAffineTransform(src_pts, dst_pts)
            warped_img = cv2.warpAffine(rotated_img, M, (256, 256), borderMode=cv2.BORDER_REPLICATE)
            
            pt_warp = np.array([rotated_pt[0], rotated_pt[1], 1.0], dtype=np.float32)
            warped_pt = np.dot(M, pt_warp)
            
            x_warped = np.clip(warped_pt[0] / 256.0, 0.0, 1.0)
            y_warped = np.clip(warped_pt[1] / 256.0, 0.0, 1.0)
            
            return warped_img, [x_warped, y_warped]
            
        def get_crop(img, T_x, T_y, W, H, img_w, img_h):
            cx = T_x + np.random.uniform(-0.3 * W, 0.3 * W)
            cy = T_y + np.random.uniform(-0.3 * H, 0.3 * H)
            
            x1 = max(0, min(int(cx - W / 2), img_w - W))
            y1 = max(0, min(int(cy - H / 2), img_h - H))
            
            crop = img[y1:y1+H, x1:x1+W]
            x_norm = (T_x - x1) / W
            y_norm = (T_y - y1) / H
            
            crop_256 = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
            return crop_256, [x_norm, y_norm]
            
        for b in range(num_batches):
            hist_frames_batch = []
            prev_frames_batch = []
            curr_frames_batch = []
            target_heatmaps_batch = []
            
            # Debug tracking coordinates
            hist_coords_batch = []
            prev_coords_batch = []
            curr_coords_batch = []
            
            for _ in range(batch_size):
                success = False
                while not success:
                    random_img_path = np.random.choice(image_paths)
                    img = cv2.imread(random_img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None or img.shape[0] < 300 or img.shape[1] < 300:
                        continue
                        
                    img_h, img_w = img.shape[:2]
                    
                    w_base = int(img_w * np.random.uniform(0.35, 0.55))
                    h_base = int(img_h * np.random.uniform(0.35, 0.55))
                    w_base = max(160, min(w_base, img_w - 60))
                    h_base = max(160, min(h_base, img_h - 60))
                    
                    cx_base = int(img_w / 2 + np.random.uniform(-0.15 * img_w, 0.15 * img_w))
                    cy_base = int(img_h / 2 + np.random.uniform(-0.15 * img_h, 0.15 * img_h))
                    
                    x1_base = max(30, cx_base - w_base // 2)
                    y1_base = max(30, cy_base - h_base // 2)
                    x2_base = min(img_w - 30, x1_base + w_base)
                    y2_base = min(img_h - 30, y1_base + h_base)
                    
                    w_base = x2_base - x1_base
                    h_base = y2_base - y1_base
                    
                    T_x = x1_base + np.random.uniform(0.15, 0.85) * w_base
                    T_y = y1_base + np.random.uniform(0.15, 0.85) * h_base
                    
                    w_hist, h_hist = w_base, h_base
                    w_prev, h_prev = int(w_base * 0.85), int(h_base * 0.85)
                    w_curr, h_curr = int(w_base * 0.7), int(h_base * 0.7)
                    
                    hist_crop, hist_norm = get_crop(img, T_x, T_y, w_hist, h_hist, img_w, img_h)
                    prev_crop, prev_norm = get_crop(img, T_x, T_y, w_prev, h_prev, img_w, img_h)
                    curr_crop, curr_norm = get_crop(img, T_x, T_y, w_curr, h_curr, img_w, img_h)
                    
                    # Apply random affine warp and orientation rotation
                    hist_crop_warped, hist_norm_warped = apply_parallax_warp(hist_crop, hist_norm)
                    prev_crop_warped, prev_norm_warped = apply_parallax_warp(prev_crop, prev_norm)
                    
                    # Apply circular masking in uint8 space centered around warped targets
                    hist_crop_masked = apply_circle_mask(hist_crop_warped, hist_norm_warped, radius=128)
                    prev_crop_masked = apply_circle_mask(prev_crop_warped, prev_norm_warped, radius=50)
                    
                    # Normalize pixel values
                    hist_frame = np.expand_dims(hist_crop_masked.astype(np.float32) / 255.0, axis=-1)
                    prev_frame = np.expand_dims(prev_crop_masked.astype(np.float32) / 255.0, axis=-1)
                    curr_frame = np.expand_dims(curr_crop.astype(np.float32) / 255.0, axis=-1)
                    
                    # Generate Gaussian target heatmap
                    target_heatmap = generate_gaussian_heatmap(curr_norm, size=64, sigma=4.0)
                    
                    hist_frames_batch.append(hist_frame)
                    prev_frames_batch.append(prev_frame)
                    curr_frames_batch.append(curr_frame)
                    target_heatmaps_batch.append(target_heatmap)
                    
                    # Save debug coordinates
                    hist_coords_batch.append(hist_norm_warped)
                    prev_coords_batch.append(prev_norm_warped)
                    curr_coords_batch.append(curr_norm)
                    
                    success = True
                    
            batch_data = {
                "inputs": [
                    np.array(hist_frames_batch, dtype=np.float32),
                    np.array(prev_frames_batch, dtype=np.float32),
                    np.array(curr_frames_batch, dtype=np.float32)
                ],
                "targets": np.array(target_heatmaps_batch, dtype=np.float32),
                "debug_coords": {
                    "hist": np.array(hist_coords_batch, dtype=np.float32),
                    "prev": np.array(prev_coords_batch, dtype=np.float32),
                    "curr": np.array(curr_coords_batch, dtype=np.float32)
                }
            }
            
            output_file = os.path.join(output_path, f"dataset_{b}.pkl")
            with open(output_file, "wb") as f:
                pickle.dump(batch_data, f)
                
            print(f"Generated and saved { (b+1)*batch_size }/{total_samples} samples -> {output_file}")
            
        print("Dataset generation completed successfully.")

    @staticmethod
    def load_dataset_from_pickles(dataset_dir, eval_pkl_num=4):
        """
        Discovers, sorts, and splits dataset pickle files into training and evaluation datasets.
        """
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")
            
        pickle_files = [f for f in os.listdir(dataset_dir) if f.startswith("dataset_") and f.endswith(".pkl")]
        if not pickle_files:
            raise FileNotFoundError(f"No pickle files found in '{dataset_dir}'.")
            
        def extract_num(filename):
            try:
                return int(filename.split("_")[1].split(".")[0])
            except Exception:
                return 999999
                
        pickle_files.sort(key=extract_num)
        
        val_files = [os.path.join(dataset_dir, f) for f in pickle_files[:eval_pkl_num]]
        train_files = [os.path.join(dataset_dir, f) for f in pickle_files[eval_pkl_num:]]
        
        if not val_files:
            raise ValueError("No validation pickles available.")
        if not train_files:
            print("Warning: No training pickles remaining. Using validation files for training.")
            train_files = val_files
            
        def make_generator(paths):
            def generator():
                for path in paths:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                    inputs = data["inputs"]
                    targets = data["targets"]
                    yield (tuple(inputs), targets)
            return generator
            
        output_signature = (
            (
                tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32, name="hist_frame"),
                tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32, name="prev_frame"),
                tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32, name="curr_frame"),
            ),
            tf.TensorSpec(shape=(None, 64, 64, 1), dtype=tf.float32, name="target_heatmap")
        )
        
        val_dataset = tf.data.Dataset.from_generator(
            make_generator(val_files),
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)
        
        train_dataset = tf.data.Dataset.from_generator(
            make_generator(train_files),
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset

    def evaluate(self, dataset, loss_fn):
        if self.model is None:
            raise ValueError("Model is not initialized.")
            
        val_loss_avg = tf.keras.metrics.Mean()
        for inputs, targets in dataset:
            predictions = self.model(inputs, training=False)
            loss_value = loss_fn(targets, predictions)
            val_loss_avg.update_state(loss_value)
            
        return float(val_loss_avg.result())

    def train_epoch(self, dataset, optimizer, loss_fn):
        epoch_loss_avg = tf.keras.metrics.Mean()
        for inputs, targets in dataset:
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                loss_value = loss_fn(targets, predictions)
                
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            epoch_loss_avg.update_state(loss_value)
            
        return float(epoch_loss_avg.result())

    def train(self, train_dataset, val_dataset, lr, num_of_epochs, loss_name="mse", output_path=None, best_train_loss_output=None):
        if loss_name == "mse":
            loss_fn = losses.MeanSquaredError()
        elif loss_name == "huber":
            loss_fn = losses.Huber(delta=1.0)
        elif loss_name == "logcosh":
            loss_fn = losses.LogCosh()
        elif loss_name == "dice_bce":
            loss_fn = dice_bce_loss
        elif loss_name == "focal":
            loss_fn = focal_loss
        else:
            raise ValueError(f"Unknown loss function for heatmap regression: {loss_name}")
            
        if self.model is None:
            raise ValueError("Model is not initialized.")
            
        optimizer = optimizers.Adam(learning_rate=lr)
        
        epsilon = 1e-12
        print("Calculating initial score on validation dataset...")
        initial_val_loss = self.evaluate(val_dataset, loss_fn)
        best_score = 1.0 / (initial_val_loss + epsilon)
        best_train_loss = float('inf')
        print(f"Initial Validation Loss: {initial_val_loss:.6f} | Initial Best Score: {best_score:.4f}")
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_score": []
        }
        
        for epoch in range(1, num_of_epochs + 1):
            epoch_loss = self.train_epoch(train_dataset, optimizer, loss_fn)
            history["train_loss"].append(epoch_loss)
            
            # Save model if training loss improved (if best_train_loss_output is configured)
            if best_train_loss_output is not None and epoch_loss < best_train_loss:
                old_train_loss = best_train_loss
                best_train_loss = epoch_loss
                print(f"   [TRAIN IMPROVEMENT] Training loss improved from {old_train_loss:.6f} to {best_train_loss:.6f}! Saving clean model to {best_train_loss_output}...")
                
                # Auto-create parent directories if they do not exist
                parent_dir = os.path.dirname(best_train_loss_output)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                    
                # In-memory weight transfer to completely strip compiled training configs
                temp_tracker = TargetTracker2(input_shape=self.input_shape)
                temp_model = temp_tracker.create_model()
                temp_model.set_weights(self.model.get_weights())
                temp_model.save(best_train_loss_output)
            
            val_loss = self.evaluate(val_dataset, loss_fn)
            epoch_score = 1.0 / (val_loss + epsilon)
            
            history["val_loss"].append(val_loss)
            history["val_score"].append(epoch_score)
            
            print(f"Epoch {epoch:03d}/{num_of_epochs:03d} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f} | Score: {epoch_score:.4f}")
            
            if epoch_score > best_score:
                old_score = best_score
                best_score = epoch_score
                
                if output_path is not None:
                    save_path = output_path
                else:
                    save_path = f"tracker2_model_score_{best_score:.4f}.keras"
                
                print(f"   [IMPROVEMENT] Score improved from {old_score:.4f} to {best_score:.4f}! Saving clean model to {save_path}...")
                
                # Auto-create parent directories if they do not exist
                parent_dir = os.path.dirname(save_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                    
                # In-memory weight transfer to completely strip compiled training configs
                temp_tracker = TargetTracker2(input_shape=self.input_shape)
                temp_model = temp_tracker.create_model()
                temp_model.set_weights(self.model.get_weights())
                temp_model.save(save_path)
                
        print("Training completed successfully.")
        return history

# =====================================================================
# Command Line Interface (CLI) Entrypoint
# =====================================================================

def main(args_list=None):
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="TargetTracker2 (FCN) CLI Tool")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available subcommands")
    
    # Subparser for generate_dataset
    gen_parser = subparsers.add_parser("generate_dataset", help="Generate a synthetic masked tracking dataset")
    gen_parser.add_argument(
        "--images_path", 
        required=True, 
        help="Path to a directory containing raw images OR path to a .txt file listing image paths (one per line)"
    )
    gen_parser.add_argument(
        "--output_path", 
        required=True, 
        help="Output directory where numbered pickle files will be saved"
    )
    gen_parser.add_argument(
        "--batch_size", 
        type=int, 
        default=256, 
        help="Batch size per pickle file (default: 256)"
    )
    gen_parser.add_argument(
        "--num_of_samples", 
        type=int, 
        default=16384, 
        help="Total number of samples to generate (default: 16384)"
    )
    
    # Subparser for train
    train_parser = subparsers.add_parser("train", help="Train the target tracker FCN model")
    train_parser.add_argument(
        "--dataset_dir", 
        required=True, 
        help="Directory containing the .pkl dataset batches"
    )
    train_parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-3, 
        help="Learning rate for optimization (default: 1e-3)"
    )
    train_parser.add_argument(
        "--num_of_epochs", 
        type=int, 
        default=10, 
        help="Number of epochs to train (default: 10)"
    )
    train_parser.add_argument(
        "--loss", 
        choices=["mse", "huber", "logcosh", "dice_bce", "focal"], 
        default="mse", 
        help="Loss function to optimize (default: mse)"
    )
    train_parser.add_argument(
        "--eval_pkl_num", 
        type=int, 
        default=4, 
        help="Number of initial pickle files to allocate for evaluation (default: 4)"
    )
    train_parser.add_argument(
        "--init_keras_file", 
        type=str, 
        default=None, 
        help="Path to initial Keras model file to resume training from"
    )
    train_parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Path to save the trained Keras model"
    )
    train_parser.add_argument(
        "--best_train_loss_output", 
        type=str, 
        default=None, 
        help="Path to save the best model based on training loss (strips training configs)"
    )
    
    args = parser.parse_args(args_list)
    
    if args.command == "generate_dataset":
        TargetTracker2.generate_dataset(
            images_path=args.images_path,
            output_path=args.output_path,
            batch_size=args.batch_size,
            num_of_samples=args.num_of_samples
        )
    elif args.command == "train":
        train_ds, val_ds = TargetTracker2.load_dataset_from_pickles(
            dataset_dir=args.dataset_dir, 
            eval_pkl_num=args.eval_pkl_num
        )
        
        tracker = TargetTracker2()
        
        if args.init_keras_file and os.path.exists(args.init_keras_file):
            import tensorflow as tf
            print(f"Resuming training: loading existing FCN model from {args.init_keras_file}...")
            tracker.model = tf.keras.models.load_model(args.init_keras_file, compile=False)
        else:
            print("No initial model file found or specified. Building new Keras FCN model...")
            tracker.create_model()
            
        tracker.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            lr=args.lr,
            num_of_epochs=args.num_of_epochs,
            loss_name=args.loss,
            output_path=args.output,
            best_train_loss_output=args.best_train_loss_output
        )

if __name__ == "__main__":
    main()
