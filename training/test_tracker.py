import os
import tensorflow as tf
import numpy as np
from tracker.model import TargetTracker

def run_tests():
    print("=== TargetTracker Automated Verification ===")
    
    # 1. Initialization
    print("1. Initializing TargetTracker...")
    tracker = TargetTracker(input_shape=(256, 256, 1), max_offset=0.2)
    print("   [SUCCESS] TargetTracker successfully initialized.")
    
    # 2. Model Creation
    print("\n2. Creating Keras Tracking Model...")
    model = tracker.create_model()
    model.summary()
    print("   [SUCCESS] Model successfully built.")
    
    # Verify input/output shapes
    assert len(model.inputs) == 5, f"Expected 5 inputs, got {len(model.inputs)}"
    assert model.output_shape == (None, 2), f"Expected output shape (None, 2), got {model.output_shape}"
    print(f"   [SUCCESS] Input count: {len(model.inputs)}, Output shape: {model.output_shape}")
    
    # 3. Dummy Forward Pass
    print("\n3. Testing forward pass with dummy tensors...")
    batch_size = 4
    dummy_hist_frame = tf.random.uniform((batch_size, 256, 256, 1), dtype=tf.float32)
    dummy_hist_coords = tf.random.uniform((batch_size, 2), minval=0.1, maxval=0.9, dtype=tf.float32)
    dummy_prev_frame = tf.random.uniform((batch_size, 256, 256, 1), dtype=tf.float32)
    dummy_prev_coords = tf.random.uniform((batch_size, 2), minval=0.1, maxval=0.9, dtype=tf.float32)
    dummy_curr_frame = tf.random.uniform((batch_size, 256, 256, 1), dtype=tf.float32)
    
    predictions = model([dummy_hist_frame, dummy_hist_coords, dummy_prev_frame, dummy_prev_coords, dummy_curr_frame])
    
    print(f"   Input prev_coords sample:\n{dummy_prev_coords.numpy()}")
    print(f"   Output new_coords sample:\n{predictions.numpy()}")
    
    assert predictions.shape == (batch_size, 2), f"Expected shape (4, 2), got {predictions.shape}"
    print("   [SUCCESS] Forward pass completed with correct output shape.")
    
    # 4. Custom Training Loop Verification
    print("\n4. Testing custom training loop (2 epochs) on dummy dataset...")
    
    # Create fake dataset of 5 batches
    num_batches = 5
    x_hist = np.random.rand(num_batches * batch_size, 256, 256, 1).astype(np.float32)
    coords_hist = np.random.uniform(0.1, 0.9, (num_batches * batch_size, 2)).astype(np.float32)
    x_prev = np.random.rand(num_batches * batch_size, 256, 256, 1).astype(np.float32)
    coords_prev = np.random.uniform(0.1, 0.9, (num_batches * batch_size, 2)).astype(np.float32)
    x_curr = np.random.rand(num_batches * batch_size, 256, 256, 1).astype(np.float32)
    coords_target = np.clip(coords_prev + np.random.uniform(-0.1, 0.1, (num_batches * batch_size, 2)), 0.0, 1.0).astype(np.float32)
    
    # Create tf.data.Dataset
    # The training loop expects: ((hist_frames, hist_coords, prev_frames, prev_coords, curr_frames), target_coords)
    train_dataset = tf.data.Dataset.from_tensor_slices((
        (x_hist, coords_hist, x_prev, coords_prev, x_curr),
        coords_target
    )).batch(batch_size)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        (x_hist, coords_hist, x_prev, coords_prev, x_curr),
        coords_target
    )).batch(batch_size)
    
    # Execute training
    history = tracker.train(train_dataset, val_dataset, lr=1e-3, num_of_epochs=2, loss_name="logcosh")
    
    assert 'train_loss' in history, "Expected history to contain train_loss"
    assert len(history['train_loss']) == 2, "Expected history to contain 2 epochs of train_loss"
    print("   [SUCCESS] Custom training epochs and evaluation run flawlessly.")
    print("=== All Verification Tests Passed Successfully! ===")

if __name__ == "__main__":
    run_tests()
