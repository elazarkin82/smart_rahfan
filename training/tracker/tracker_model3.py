import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import os
import cv2
import pickle
import numpy as np

# =====================================================================
# Helper Functions for Soft/Circular Masking and Gaussian Heatmaps
# =====================================================================

def generate_circular_mask(coords, size=256, radius=50):
    """
    Generates a binary circular attention mask of shape (size, size, 1)
    centered around normalized coordinates.
    
    Args:
        coords (list/tuple): Normalized [x, y] coordinates in [0, 1] range.
        size (int): Dimensions of the output mask (default: 256).
        radius (int): Radius of the circular mask in pixels (default: 50).
        
    Returns:
        np.ndarray: Binary circular mask array of shape (size, size, 1).
    """
    mask = np.zeros((size, size), dtype=np.float32)
    x_px = int(coords[0] * size)
    y_px = int(coords[1] * size)
    cv2.circle(mask, (x_px, y_px), radius, 1.0, -1)
    return np.expand_dims(mask, axis=-1)

def generate_gaussian_mask(coords, size=256, sigma=30.0):
    """
    Generates a 2D Gaussian soft attention mask of shape (size, size, 1)
    centered around normalized coordinates to avoid sharp artificial edges.
    
    Args:
        coords (list/tuple): Normalized [x, y] coordinates in [0, 1] range.
        size (int): Dimensions of the output mask (default: 256).
        sigma (float): Standard deviation of the Gaussian kernel (default: 30.0).
        
    Returns:
        np.ndarray: Gaussian soft mask array of shape (size, size, 1).
    """
    x_target = coords[0] * size
    y_target = coords[1] * size
    
    x = np.arange(0, size, 1, dtype=np.float32)
    y = np.arange(0, size, 1, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Compute squared Euclidean distance from target point
    d2 = (x_grid - x_target) ** 2 + (y_grid - y_target) ** 2
    # Apply standard 2D Gaussian formula
    mask = np.exp(-d2 / (2.0 * sigma ** 2))
    
    return np.expand_dims(mask, axis=-1)

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
# Fully Convolutional Target Tracker Class (V3 - Multi-Channel Input)
# =====================================================================

class TargetTracker3:
    """
    A Fully Convolutional recursive target tracker in Keras (V3).
    
    To retain global scene context (essential for learning side-motion/translation cues)
    while guiding model focus, this model:
      - Keeps input frames COMPLETELY UNMASKED (preserves full peripheral background).
      - Utilizes 2-channel inputs for each Siamese branch:
        Channel 0: The raw grayscale image frame (256, 256, 1)
        Channel 1: A spatial attention mask (256, 256, 1) representing prior target location
      - The Siamese branches:
        1. hist_input: (256, 256, 2) -> [hist_frame_unmasked, hist_soft_mask]
        2. prev_input: (256, 256, 2) -> [prev_frame_unmasked, prev_soft_mask]
        3. curr_input: (256, 256, 2) -> [curr_frame_unmasked, zero_mask]
      - Output: Regresses a spatial probability distribution (Gaussian Heatmap) of shape (64, 64, 1).
    """
    def __init__(self, input_shape=(256, 256, 2)):
        self.input_shape = input_shape
        self.model = None
        
    def _create_cnn_backbone(self):
        """
        Creates a shared Siamese CNN backbone for feature extraction.
        Accepts 2-channel inputs: (256, 256, 2)
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
        # Inputs: 3 branches, each taking a 2-channel stacked input (Image + Mask)
        hist_input = layers.Input(shape=self.input_shape, name="hist_input")
        prev_input = layers.Input(shape=self.input_shape, name="prev_input")
        curr_input = layers.Input(shape=self.input_shape, name="curr_input")
        
        # Shared Siamese Backbone (accepts 2-channel inputs)
        cnn_backbone = self._create_cnn_backbone()
        
        # Extract spatial features at (16, 16, 256)
        hist_features = cnn_backbone(hist_input)
        prev_features = cnn_backbone(prev_input)
        curr_features = cnn_backbone(curr_input)
        
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
            inputs=[hist_input, prev_input, curr_input],
            outputs=output_heatmap,
            name="TargetTrackerFCN3"
        )
        
        return self.model

    @staticmethod
    def generate_dataset(*args, **kwargs):
        """
        [TODO] Dataset generation has been decoupled and moved to a separate dedicated project.
        This placeholder is kept for future integration or compatibility.
        """
        raise NotImplementedError(
            "Dataset generation is now managed in a separate dedicated project. "
            "Please use the external dataset generator project to produce 2-channel inputs."
        )

    @staticmethod
    def load_dataset_from_pickles(dataset_dir, eval_pkl_num=4):
        """
        Discovers, sorts, and splits dataset pickle files into training and evaluation datasets.
        Expects pickles to contain 2-channel stacked inputs: shape (None, 256, 256, 2).
        """
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")
            
        # 1. Non-recursively find all files with .pkl or .pickle extensions
        pickle_files = []
        for f in os.listdir(dataset_dir):
            file_path = os.path.join(dataset_dir, f)
            if os.path.isfile(file_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in ('.pkl', '.pickle'):
                    pickle_files.append(f)
                    
        if not pickle_files:
            raise FileNotFoundError(
                f"No .pkl or .pickle files found directly in '{dataset_dir}'. "
                f"Note: Subdirectories were intentionally ignored."
            )
            
        # 2. Sort lexicographically to ensure stable and deterministic split
        pickle_files.sort()
        
        val_files = [os.path.join(dataset_dir, f) for f in pickle_files[:eval_pkl_num]]
        train_files = [os.path.join(dataset_dir, f) for f in pickle_files[eval_pkl_num:]]
        
        if not val_files:
            raise ValueError("No validation pickles available.")
        if not train_files:
            print("Warning: No training pickles remaining. Using validation files for training.")
            train_files = val_files
            
        # 3. Dynamic generator that supports per-epoch shuffling of training files
        def make_generator(paths, shuffle=False):
            def generator():
                local_paths = list(paths)
                if shuffle:
                    import random
                    random.shuffle(local_paths)
                for path in local_paths:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                    inputs = data["inputs"]
                    targets = data["targets"]
                    yield (tuple(inputs), targets)
            return generator
            
        output_signature = (
            (
                tf.TensorSpec(shape=(None, 256, 256, 2), dtype=tf.float32, name="hist_input"),
                tf.TensorSpec(shape=(None, 256, 256, 2), dtype=tf.float32, name="prev_input"),
                tf.TensorSpec(shape=(None, 256, 256, 2), dtype=tf.float32, name="curr_input"),
            ),
            tf.TensorSpec(shape=(None, 64, 64, 1), dtype=tf.float32, name="target_heatmap")
        )
        
        val_dataset = tf.data.Dataset.from_generator(
            make_generator(val_files, shuffle=False),
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)
        val_dataset.steps = len(val_files)
        
        train_dataset = tf.data.Dataset.from_generator(
            make_generator(train_files, shuffle=True),
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)
        train_dataset.steps = len(train_files)
        
        return train_dataset, val_dataset

    def log(self, message, log_file=None):
        """
        Prints a clean message to the console using tqdm.write to avoid breaking
        active progress bars, and appends it to a log file if specified.
        """
        from tqdm import tqdm
        tqdm.write(message)
        
        if log_file:
            try:
                parent_dir = os.path.dirname(log_file)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(message + "\n")
            except Exception as e:
                tqdm.write(f"Warning: Failed to write to log file '{log_file}': {e}")

    def evaluate(self, dataset, loss_fn):
        if self.model is None:
            raise ValueError("Model is not initialized.")
            
        val_loss_avg = tf.keras.metrics.Mean()
        steps = getattr(dataset, "steps", None)
        
        from tqdm import tqdm
        progress_bar = tqdm(dataset, total=steps, desc="Evaluating", leave=False)
        
        for inputs, targets in progress_bar:
            predictions = self.model(inputs, training=False)
            loss_value = loss_fn(targets, predictions)
            val_loss_avg.update_state(loss_value)
            
        return float(val_loss_avg.result())

    def train_epoch(self, dataset, optimizer, loss_fn, epoch, num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        steps = getattr(dataset, "steps", None)
        
        from tqdm import tqdm
        progress_bar = tqdm(
            dataset, 
            total=steps, 
            desc=f"Epoch {epoch:03d}/{num_epochs:03d}", 
            leave=False
        )
        
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
        self.log("Calculating initial score on validation dataset...", log_file)
        initial_val_loss = self.evaluate(val_dataset, loss_fn)
        best_score = 1.0 / (initial_val_loss + epsilon)
        best_train_loss = float('inf')
        self.log(f"Initial Validation Loss: {initial_val_loss:.6f} | Initial Best Score: {best_score:.4f}", log_file)
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_score": []
        }
        
        for epoch in range(1, num_of_epochs + 1):
            epoch_loss = self.train_epoch(train_dataset, optimizer, loss_fn, epoch, num_of_epochs)
            history["train_loss"].append(epoch_loss)
            
            # Save model if training loss improved (if best_train_loss_output is configured)
            if best_train_loss_output is not None and epoch_loss < best_train_loss:
                old_train_loss = best_train_loss
                best_train_loss = epoch_loss
                self.log(f"   [TRAIN IMPROVEMENT] Training loss improved from {old_train_loss:.6f} to {best_train_loss:.6f}! Saving clean model to {best_train_loss_output}...", log_file)
                
                # Auto-create parent directories if they do not exist
                parent_dir = os.path.dirname(best_train_loss_output)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                    
                # In-memory weight transfer to completely strip compiled training configs
                temp_tracker = TargetTracker3(input_shape=self.input_shape)
                temp_model = temp_tracker.create_model()
                temp_model.set_weights(self.model.get_weights())
                temp_model.save(best_train_loss_output)
            
            val_loss = self.evaluate(val_dataset, loss_fn)
            epoch_score = 1.0 / (val_loss + epsilon)
            
            history["val_loss"].append(val_loss)
            history["val_score"].append(epoch_score)
            
            self.log(f"Epoch {epoch:03d}/{num_of_epochs:03d} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f} | Score: {epoch_score:.4f}", log_file)
            
            if epoch_score > best_score:
                old_score = best_score
                best_score = epoch_score
                
                if output_path is not None:
                    save_path = output_path
                else:
                    save_path = f"tracker3_model_score_{best_score:.4f}.keras"
                
                self.log(f"   [IMPROVEMENT] Score improved from {old_score:.4f} to {best_score:.4f}! Saving clean model to {save_path}...", log_file)
                
                # Auto-create parent directories if they do not exist
                parent_dir = os.path.dirname(save_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                    
                # In-memory weight transfer to completely strip compiled training configs
                temp_tracker = TargetTracker3(input_shape=self.input_shape)
                temp_model = temp_tracker.create_model()
                temp_model.set_weights(self.model.get_weights())
                temp_model.save(save_path)
                
        self.log("Training completed successfully.", log_file)
        return history

# =====================================================================
# Command Line Interface (CLI) Entrypoint
# =====================================================================

def main(args_list=None):
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="TargetTracker3 (FCN Multi-Channel) CLI Tool")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available subcommands")
    
    # Subparser for generate_dataset (Kept as placeholder stub)
    gen_parser = subparsers.add_parser("generate_dataset", help="Dataset generation is managed by a separate project")
    
    # Subparser for train
    train_parser = subparsers.add_parser("train", help="Train the target tracker FCN model with 2-channel inputs")
    train_parser.add_argument(
        "--dataset_dir", 
        required=True, 
        help="Directory containing the .pkl 2-channel dataset batches"
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
    train_parser.add_argument(
        "--log_file", 
        type=str, 
        default=None, 
        help="Path to file where clean training logs will be written"
    )
    
    args = parser.parse_args(args_list)
    
    if args.command == "generate_dataset":
        tracker = TargetTracker3()
        tracker.generate_dataset()
    elif args.command == "train":
        train_ds, val_ds = TargetTracker3.load_dataset_from_pickles(
            dataset_dir=args.dataset_dir, 
            eval_pkl_num=args.eval_pkl_num
        )
        
        tracker = TargetTracker3()
        
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
            best_train_loss_output=args.best_train_loss_output,
            log_file=args.log_file
        )

if __name__ == "__main__":
    main()
