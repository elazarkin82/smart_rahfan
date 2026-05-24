import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

class TargetTracker:
    """
    A recursive deep-learning-based target tracker in Keras.
    
    This model tracks a target object across consecutive video frames. It takes:
      1. Distant (historical) frame: (256, 256, 1) grayscale image
      2. Distant (historical) target coordinates: [x, y] in normalized [0, 1] range
      3. Previous frame: (256, 256, 1) grayscale image
      4. Previous target coordinates: [x, y] in normalized [0, 1] range
      5. Current frame: (256, 256, 1) grayscale image
    
    It outputs:
      - Predicted new target coordinates [x, y] in the current frame.
      
    To achieve high recursive stability and prevent drift over time, the model:
      - Uses a shared Siamese CNN to learn frame-invariant target appearance across all three frames.
      - Projects both historical and previous coordinates to spatial grids, injecting them at the 
        CNN bottlenecks to merge visual content with corresponding location priors.
      - Regresses a relative coordinate offset (bounded by a maximum displacement) relative to the 
        previous target position, ensuring local consistency and preventing sudden drift.
    """
    
    def __init__(self, input_shape=(256, 256, 1), max_offset=0.2):
        """
        Initializes the tracker.
        
        Args:
            input_shape (tuple): Shape of the input frames (H, W, C).
            max_offset (float): Maximum normalized step (offset) the target can move between frames.
        """
        self.input_shape = input_shape
        self.max_offset = max_offset
        self.model = None
        
    def _create_cnn_backbone(self):
        """
        Creates a shared CNN backbone for feature extraction.
        
        Returns:
            tf.keras.Model: The CNN backbone model.
        """
        inputs = layers.Input(shape=self.input_shape, name="cnn_input")
        
        # Stride-2 convolutions progressively downsample HxW while increasing channel depth
        x = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(64, (3, 3), strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(128, (3, 3), strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(128, (3, 3), strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(256, (3, 3), strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        # Output shape: (8, 8, 256)
        
        return models.Model(inputs, x, name="siamese_cnn_backbone")

    def create_model(self):
        """
        Builds and compiles the Keras functional tracking model.
        
        Returns:
            tf.keras.Model: The constructed Keras tracking model.
        """
        # Inputs
        hist_frame_in = layers.Input(shape=self.input_shape, name="hist_frame")
        hist_coords_in = layers.Input(shape=(2,), name="hist_coords") # [x, y] normalized
        
        prev_frame_in = layers.Input(shape=self.input_shape, name="prev_frame")
        prev_coords_in = layers.Input(shape=(2,), name="prev_coords") # [x, y] normalized
        
        curr_frame_in = layers.Input(shape=self.input_shape, name="curr_frame")
        
        # Instantiate Siamese CNN Backbone
        cnn_backbone = self._create_cnn_backbone()
        
        # Extract features from all three frames
        hist_features = cnn_backbone(hist_frame_in) # Shape: (8, 8, 256)
        prev_features = cnn_backbone(prev_frame_in) # Shape: (8, 8, 256)
        curr_features = cnn_backbone(curr_frame_in) # Shape: (8, 8, 256)
        
        # Spatial Coordinate Injection for Historical Frame:
        hist_coord_proj = layers.Dense(8 * 8 * 16, activation="relu", name="hist_coords_projection")(hist_coords_in)
        hist_coord_grid = layers.Reshape((8, 8, 16), name="hist_coords_reshaped")(hist_coord_proj)
        hist_features_with_coords = layers.Concatenate(axis=-1, name="hist_spatial_fusion")(
            [hist_features, hist_coord_grid]
        ) # Shape: (8, 8, 272)
        
        hist_features_fused = layers.Conv2D(256, (1, 1), padding="same", activation="relu", name="hist_bottleneck")(
            hist_features_with_coords
        )
        hist_features_fused = layers.BatchNormalization()(hist_features_fused)
        
        # Spatial Coordinate Injection for Previous Frame:
        prev_coord_proj = layers.Dense(8 * 8 * 16, activation="relu", name="prev_coords_projection")(prev_coords_in)
        prev_coord_grid = layers.Reshape((8, 8, 16), name="prev_coords_reshaped")(prev_coord_proj)
        prev_features_with_coords = layers.Concatenate(axis=-1, name="prev_spatial_fusion")(
            [prev_features, prev_coord_grid]
        ) # Shape: (8, 8, 272)
        
        prev_features_fused = layers.Conv2D(256, (1, 1), padding="same", activation="relu", name="prev_bottleneck")(
            prev_features_with_coords
        )
        prev_features_fused = layers.BatchNormalization()(prev_features_fused)
        
        # Temporal Feature Fusion & Motion Dynamics:
        # We calculate displacements: hist -> prev, and prev -> curr
        motion_hist_to_prev = layers.Subtract(name="visual_motion_hist_to_prev")([prev_features_fused, hist_features_fused])
        motion_prev_to_curr = layers.Subtract(name="visual_motion_prev_to_curr")([curr_features, prev_features_fused])
        
        # Concatenate fused visual features and motion dynamics to handle geometric transformations
        fused_features = layers.Concatenate(axis=-1, name="temporal_visual_fusion")(
            [hist_features_fused, prev_features_fused, curr_features, motion_hist_to_prev, motion_prev_to_curr]
        ) # Shape: (8, 8, 1280)
        
        # Process fused representations
        x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="displacement_conv")(fused_features)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D(name="displacement_pooling")(x) # Shape: (256,)
        
        # Regressor Network
        x = layers.Dense(128, activation="relu", name="fc_regressor_1")(x)
        x = layers.Dropout(0.2, name="dropout_regressor")(x)
        x = layers.Dense(64, activation="relu", name="fc_regressor_2")(x)
        
        # Regression Layer - outputs delta coordinates relative to previous frame position
        delta_coords = layers.Dense(2, activation="tanh", name="delta_coords_raw")(x)
        
        # Scale displacement by max_offset to constrain step size and maximize recursive stability
        scaled_delta = layers.Lambda(
            lambda d: d * self.max_offset, 
            name="delta_coords_scaled"
        )(delta_coords)
        
        # Final absolute coordinates = previous coordinates + scaled displacement
        new_coords = layers.Add(name="new_coords")([prev_coords_in, scaled_delta])
        
        # Define functional Keras model
        self.model = models.Model(
            inputs=[hist_frame_in, hist_coords_in, prev_frame_in, prev_coords_in, curr_frame_in],
            outputs=new_coords,
            name="TargetTrackerNetwork"
        )
        
        return self.model
        
    @staticmethod
    def generate_dataset(images_path, output_path, batch_size=256, num_of_samples=16384):
        """
        Generates a synthetic tracking dataset from a directory of raw images or a text file listing images.
        
        This method processes raw images and extracts sequences representing:
          1. hist_frame: A distant view (larger bounding box).
          2. hist_coords: The [x, y] coordinates of the target in the distant view.
          3. prev_frame: An immediate previous view (medium bounding box).
          4. prev_coords: The [x, y] coordinates of the target in the previous view.
          5. curr_frame: A current view (smaller bounding box, representing zooming in).
        
        The targets are the [x, y] coordinates of the target in the current frame.
        Both hist_frame and prev_frame are warped with random affine transformations
        to simulate real-world parallax and perspective distortions. The target
        coordinates are mathematically transformed by the same matrix to ensure label alignment.
        
        Args:
            images_path (str): Path to a directory containing raw images OR path to a .txt file
                               containing absolute/relative image paths (one per line).
            output_path (str): Output directory where the pickle files will be saved.
            batch_size (int): Batch size per pickle file.
            num_of_samples (int): Desired total number of samples. This will be rounded
                                  up to the nearest multiple of batch_size.
        """
        import os
        import cv2
        import pickle
        import numpy as np
        
        print("Starting dataset generation crawling...")
        
        image_extensions = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")
        image_paths = []
        
        if os.path.isdir(images_path):
            # Crawl directory recursively
            for root, _, files in os.walk(images_path):
                for f in files:
                    if f.lower().endswith(image_extensions):
                        image_paths.append(os.path.join(root, f))
            if not image_paths:
                raise ValueError(f"No images with extensions {image_extensions} found in directory {images_path}")
        elif os.path.isfile(images_path):
            # Read from text file list
            with open(images_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                path = line.strip()
                if path and path.lower().endswith(image_extensions):
                    if os.path.exists(path):
                        image_paths.append(path)
                    else:
                        print(f"Warning: Image file listed in txt does not exist: {path}")
            if not image_paths:
                raise ValueError(f"No valid existing images listed in text file {images_path}")
        else:
            raise ValueError(f"Invalid images_path (not a directory or file): {images_path}")
            
        print(f"Found {len(image_paths)} valid images. Preparing samples...")
        
        # 2. Round num_of_samples to be divisible by batch_size
        num_batches = int(np.ceil(num_of_samples / batch_size))
        total_samples = num_batches * batch_size
        print(f"Requested {num_of_samples} samples. Adjusted to {total_samples} samples ({num_batches} batches of size {batch_size}).")
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Helper to apply random affine warp and rotation for parallax simulation
        def apply_parallax_warp(img_256, coords):
            # 1. Random Orientation Rotation (between -20.0 and +20.0 degrees)
            angle = np.random.uniform(-20.0, 20.0)
            R = cv2.getRotationMatrix2D((128.0, 128.0), angle, 1.0)
            rotated_img = cv2.warpAffine(img_256, R, (256, 256), borderMode=cv2.BORDER_REPLICATE)
            
            # Map normalized target coordinates to 256x256 pixel space and apply rotation
            x_px, y_px = coords[0] * 256.0, coords[1] * 256.0
            pt_rot = np.array([x_px, y_px, 1.0], dtype=np.float32)
            rotated_pt = np.dot(R, pt_rot)
            
            # 2. Strong Affine Distortion (Corner perturbations up to 25 pixels)
            src_pts = np.float32([[15, 15], [240, 15], [15, 240]])
            delta = 25.0  # Increased for highly visible, strong geometric warping
            dst_pts = src_pts + np.random.uniform(-delta, delta, src_pts.shape).astype(np.float32)
            
            M = cv2.getAffineTransform(src_pts, dst_pts)
            warped_img = cv2.warpAffine(rotated_img, M, (256, 256), borderMode=cv2.BORDER_REPLICATE)
            
            # Apply affine warp to the already rotated coordinates
            pt_warp = np.array([rotated_pt[0], rotated_pt[1], 1.0], dtype=np.float32)
            warped_pt = np.dot(M, pt_warp)
            
            # Re-normalize and clip to [0, 1]
            x_warped = np.clip(warped_pt[0] / 256.0, 0.0, 1.0)
            y_warped = np.clip(warped_pt[1] / 256.0, 0.0, 1.0)
            
            return warped_img, [x_warped, y_warped]
            
        # Helper to crop a window around target point
        def get_crop(img, T_x, T_y, W, H, img_w, img_h):
            # Center of the crop is T_x, T_y with some random lateral offset (drift simulation)
            cx = T_x + np.random.uniform(-0.3 * W, 0.3 * W)
            cy = T_y + np.random.uniform(-0.3 * H, 0.3 * H)
            
            # Window boundaries
            x1 = int(cx - W / 2)
            y1 = int(cy - H / 2)
            
            # Clip to image boundaries
            x1 = max(0, min(x1, img_w - W))
            y1 = max(0, min(y1, img_h - H))
            x2 = x1 + W
            y2 = y1 + H
            
            # Extract crop
            crop = img[y1:y2, x1:x2]
            
            # Target position relative to this crop (normalized)
            x_norm = (T_x - x1) / W
            y_norm = (T_y - y1) / H
            
            # Resize crop to 256x256
            crop_256 = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
            
            return crop_256, [x_norm, y_norm]
            
        # Generate samples batch by batch
        sample_idx = 0
        for b in range(num_batches):
            hist_frames_batch = []
            hist_coords_batch = []
            prev_frames_batch = []
            prev_coords_batch = []
            curr_frames_batch = []
            curr_coords_batch = []
            
            for _ in range(batch_size):
                success = False
                while not success:
                    # Choose a random image
                    random_img_path = np.random.choice(image_paths)
                    img = cv2.imread(random_img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None or img.shape[0] < 300 or img.shape[1] < 300:
                        continue  # Skip invalid or too small images
                        
                    img_h, img_w = img.shape[:2]
                    
                    # Define base crop dimensions (roughly 35% to 55% of the image)
                    w_base = int(img_w * np.random.uniform(0.35, 0.55))
                    h_base = int(img_h * np.random.uniform(0.35, 0.55))
                    
                    # Keep aspect ratio bounded and crop sizes realistic
                    w_base = max(160, min(w_base, img_w - 60))
                    h_base = max(160, min(h_base, img_h - 60))
                    
                    # Center of base area (around center of image with padding)
                    cx_base = int(img_w / 2 + np.random.uniform(-0.15 * img_w, 0.15 * img_w))
                    cy_base = int(img_h / 2 + np.random.uniform(-0.15 * img_h, 0.15 * img_h))
                    
                    x1_base = max(30, cx_base - w_base // 2)
                    y1_base = max(30, cy_base - h_base // 2)
                    x2_base = min(img_w - 30, x1_base + w_base)
                    y2_base = min(img_h - 30, y1_base + h_base)
                    
                    w_base = x2_base - x1_base
                    h_base = y2_base - y1_base
                    
                    # Choose target point T_x, T_y inside this base crop
                    T_x = x1_base + np.random.uniform(0.15, 0.85) * w_base
                    T_y = y1_base + np.random.uniform(0.15, 0.85) * h_base
                    
                    # Sizes for all three frames (hist > prev > curr)
                    w_hist, h_hist = w_base, h_base
                    w_prev, h_prev = int(w_base * 0.85), int(h_base * 0.85)
                    w_curr, h_curr = int(w_base * 0.7), int(h_base * 0.7)
                    
                    # Crop frames and extract target coordinates
                    hist_crop, hist_norm = get_crop(img, T_x, T_y, w_hist, h_hist, img_w, img_h)
                    prev_crop, prev_norm = get_crop(img, T_x, T_y, w_prev, h_prev, img_w, img_h)
                    curr_crop, curr_norm = get_crop(img, T_x, T_y, w_curr, h_curr, img_w, img_h)
                    
                    # Apply random affine warp to hist and prev frames for parallax distortion
                    hist_crop_warped, hist_norm_warped = apply_parallax_warp(hist_crop, hist_norm)
                    prev_crop_warped, prev_norm_warped = apply_parallax_warp(prev_crop, prev_norm)
                    
                    # Reshape to (256, 256, 1) and normalize pixel values to [0, 1]
                    hist_frame = np.expand_dims(hist_crop_warped.astype(np.float32) / 255.0, axis=-1)
                    prev_frame = np.expand_dims(prev_crop_warped.astype(np.float32) / 255.0, axis=-1)
                    curr_frame = np.expand_dims(curr_crop.astype(np.float32) / 255.0, axis=-1)
                    
                    # Add to batches
                    hist_frames_batch.append(hist_frame)
                    hist_coords_batch.append(hist_norm_warped)
                    prev_frames_batch.append(prev_frame)
                    prev_coords_batch.append(prev_norm_warped)
                    curr_frames_batch.append(curr_frame)
                    curr_coords_batch.append(curr_norm)
                    
                    success = True
                    
            # Package and serialize batch
            batch_data = {
                "inputs": [
                    np.array(hist_frames_batch, dtype=np.float32),
                    np.array(hist_coords_batch, dtype=np.float32),
                    np.array(prev_frames_batch, dtype=np.float32),
                    np.array(prev_coords_batch, dtype=np.float32),
                    np.array(curr_frames_batch, dtype=np.float32)
                ],
                "targets": np.array(curr_coords_batch, dtype=np.float32)
            }
            
            output_file = os.path.join(output_path, f"dataset_{b}.pkl")
            with open(output_file, "wb") as f:
                pickle.dump(batch_data, f)
                
            sample_idx += batch_size
            print(f"Generated and saved {sample_idx}/{total_samples} samples -> {output_file}")
            
        print(f"Dataset generation completed. All pickles saved to {output_path}")
        
    @staticmethod
    def load_dataset_from_pickles(dataset_dir, eval_pkl_num=4):
        """
        Discovers, sorts, and splits dataset pickle files into training and evaluation datasets.
        
        Args:
            dataset_dir (str): Path to folder containing dataset pickle files.
            eval_pkl_num (int): Number of initial pickles to allocate for evaluation (validation).
            
        Returns:
            tuple: (train_dataset, val_dataset) as tf.data.Dataset instances.
        """
        import os
        import pickle
        import tensorflow as tf
        
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")
            
        # Discover and sort pickles numerically
        pickle_files = []
        for f in os.listdir(dataset_dir):
            if f.startswith("dataset_") and f.endswith(".pkl"):
                pickle_files.append(f)
                
        if not pickle_files:
            raise FileNotFoundError(f"No pickle files (dataset_*.pkl) found in '{dataset_dir}'.")
            
        # Sort numerically: dataset_0.pkl, dataset_1.pkl, ... dataset_63.pkl
        def extract_num(filename):
            try:
                return int(filename.split("_")[1].split(".")[0])
            except Exception:
                return 999999
                
        pickle_files.sort(key=extract_num)
        print(f"Discovered {len(pickle_files)} dataset pickle files.")
        
        # Split pickles
        val_files = [os.path.join(dataset_dir, f) for f in pickle_files[:eval_pkl_num]]
        train_files = [os.path.join(dataset_dir, f) for f in pickle_files[eval_pkl_num:]]
        
        if not val_files:
            raise ValueError(f"No validation pickle files. Adjust eval_pkl_num ({eval_pkl_num}) or add pickles.")
        if not train_files:
            print("Warning: No training pickles left after validation split! Using validation pickles for training too.")
            train_files = val_files
            
        print(f"Split: {len(val_files)} pickles for evaluation, {len(train_files)} pickles for training.")
        
        # Generator creator
        def make_generator(paths):
            def generator():
                for path in paths:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                    inputs = data["inputs"]
                    targets = data["targets"]
                    # Yields a tuple of (inputs, targets)
                    yield (tuple(inputs), targets)
            return generator
            
        # Output signature specs
        output_signature = (
            (
                tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32, name="hist_frame"),
                tf.TensorSpec(shape=(None, 2), dtype=tf.float32, name="hist_coords"),
                tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32, name="prev_frame"),
                tf.TensorSpec(shape=(None, 2), dtype=tf.float32, name="prev_coords"),
                tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32, name="curr_frame"),
            ),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32, name="target_coords")
        )
        
        # Construct tf.data.Dataset
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
        """
        Evaluates the model on the provided validation dataset.
        
        Args:
            dataset: The validation dataset (tf.data.Dataset)
            loss_fn: The loss function to evaluate with.
            
        Returns:
            float: Average loss value on the validation dataset.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call create_model() or load a model first.")
            
        val_loss_avg = tf.keras.metrics.Mean()
        
        for inputs, targets in dataset:
            predictions = self.model(inputs, training=False)
            loss_value = loss_fn(targets, predictions)
            val_loss_avg.update_state(loss_value)
            
        return float(val_loss_avg.result())

    def train_epoch(self, dataset, optimizer, loss_fn):
        """
        Trains the model for a single epoch using a custom GradientTape loop.
        
        Args:
            dataset: A tf.data.Dataset yielding a tuple:
                     ((hist_frames, hist_coords, prev_frames, prev_coords, curr_frames), target_coords)
            optimizer: A tf.keras.optimizers.Optimizer instance.
            loss_fn: A tf.keras.losses.Loss instance.
            
        Returns:
            float: Average training loss for the epoch.
        """
        epoch_loss_avg = tf.keras.metrics.Mean()
        
        for step, (inputs, targets) in enumerate(dataset):
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                loss_value = loss_fn(targets, predictions)
            
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            epoch_loss_avg.update_state(loss_value)
            
        return float(epoch_loss_avg.result())
        
    def train(self, train_dataset, val_dataset, lr, num_of_epochs, loss_name="logcosh", output_path=None):
        """
        Executes the main training process.
        
        It alternates between running a single training epoch via train_epoch
        and performing evaluation via evaluate. It implements score tracking
        and conditional model saving.
        
        Args:
            train_dataset: The training dataset (tf.data.Dataset).
            val_dataset: The validation dataset (tf.data.Dataset).
            lr (float): Learning rate for training.
            num_of_epochs (int): Number of epochs to train.
            loss_name (str): Selected loss function ('mse', 'huber', 'logcosh', 'wing').
            output_path (str, optional): Target file to save the best model.
            
        Returns:
            dict: Dictionary with lists of training and validation metrics.
        """
        # Determine the loss function
        if loss_name == "mse":
            loss_fn = losses.MeanSquaredError()
        elif loss_name == "huber":
            loss_fn = losses.Huber(delta=1.0)
        elif loss_name == "logcosh":
            loss_fn = losses.LogCosh()
        elif loss_name == "wing":
            # Custom Wing Loss for Coordinate Regression
            def wing_loss(y_true, y_pred, w=0.1, epsilon=0.01):
                diff = tf.abs(y_true - y_pred)
                C = w - w * tf.math.log(1.0 + w / epsilon)
                loss = tf.where(
                    diff < w,
                    w * tf.math.log(1.0 + diff / epsilon),
                    diff - C
                )
                return tf.reduce_mean(loss)
            loss_fn = wing_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
            
        if self.model is None:
            raise ValueError("Model is not initialized. Call create_model() or load a model first.")
            
        optimizer = optimizers.Adam(learning_rate=lr)
        
        # Calculate initial score before training
        epsilon = 1e-12
        print("Calculating initial score on validation dataset...")
        initial_val_loss = self.evaluate(val_dataset, loss_fn)
        best_score = 1.0 / (initial_val_loss + epsilon)
        print(f"Initial Validation Loss ({loss_name}): {initial_val_loss:.6f} | Initial Best Score: {best_score:.4f}")
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_score": []
        }
        
        print(f"Starting target tracker training: {num_of_epochs} epochs, lr={lr}, loss={loss_name}...")
        
        for epoch in range(1, num_of_epochs + 1):
            # Train for one epoch
            epoch_loss = self.train_epoch(train_dataset, optimizer, loss_fn)
            history["train_loss"].append(epoch_loss)
            
            # Evaluate model
            val_loss = self.evaluate(val_dataset, loss_fn)
            epoch_score = 1.0 / (val_loss + epsilon)
            
            history["val_loss"].append(val_loss)
            history["val_score"].append(epoch_score)
            
            print(f"Epoch {epoch:03d}/{num_of_epochs:03d} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f} | Score: {epoch_score:.4f}")
            
            # Save model if score improved
            if epoch_score > best_score:
                old_score = best_score
                best_score = epoch_score
                
                # Determine save path
                if output_path is not None:
                    save_path = output_path
                else:
                    save_path = f"tracker_model_score_{best_score:.4f}.keras"
                
                print(f"   [IMPROVEMENT] Score improved from {old_score:.4f} to {best_score:.4f}! Saving model to {save_path}...")
                self.model.save(save_path)
                
        print("Training completed successfully.")
        return history

def main(args_list=None):
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="TargetTracker CLI Tool")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available subcommands")
    
    # Subparser for generate_dataset
    gen_parser = subparsers.add_parser("generate_dataset", help="Generate a synthetic tracking dataset")
    gen_parser.add_argument(
        "--images_path", 
        required=True, 
        help="Path to a directory containing raw images OR path to a .txt file containing image paths (one per line)"
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
    train_parser = subparsers.add_parser("train", help="Train the target tracker model")
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
        choices=["mse", "huber", "logcosh", "wing"], 
        default="logcosh", 
        help="Loss function to optimize (default: logcosh)"
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
        help="Path to save the trained Keras model (if not defined, saves with score in filename)"
    )
    
    args = parser.parse_args(args_list)
    
    if args.command == "generate_dataset":
        TargetTracker.generate_dataset(
            images_path=args.images_path,
            output_path=args.output_path,
            batch_size=args.batch_size,
            num_of_samples=args.num_of_samples
        )
    elif args.command == "train":
        # Load and split pickles
        train_ds, val_ds = TargetTracker.load_dataset_from_pickles(
            dataset_dir=args.dataset_dir, 
            eval_pkl_num=args.eval_pkl_num
        )
        
        # Instantiate tracker
        tracker = TargetTracker()
        
        # Resume or build new model
        if args.init_keras_file and os.path.exists(args.init_keras_file):
            import tensorflow as tf
            print(f"Resuming training: loading existing model from {args.init_keras_file}...")
            tracker.model = tf.keras.models.load_model(args.init_keras_file, compile=False)
        else:
            print("No initial model file found or specified. Building new Keras model...")
            tracker.create_model()
            
        # Start training
        tracker.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            lr=args.lr,
            num_of_epochs=args.num_of_epochs,
            loss_name=args.loss,
            output_path=args.output
        )

if __name__ == "__main__":
    main()
