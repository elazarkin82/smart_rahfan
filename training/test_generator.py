import os
import shutil
import pickle
import numpy as np
import cv2
from tracker.model import TargetTracker

def run_generator_tests():
    print("=== TargetTracker Dataset Generator Verification ===")
    
    # 1. Setup Temporary Directories
    test_img_dir = "temp_test_images"
    test_output_dir = "temp_output_pickles"
    
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    print("1. Creating dummy test images...")
    # Create two dummy images of size 500x500 with simple shapes
    for i in range(2):
        img = np.zeros((500, 500), dtype=np.uint8)
        # Draw a square target in the center
        cv2.rectangle(img, (200, 200), (300, 300), 255, -1)
        # Draw some noise shapes
        cv2.circle(img, (100, 100), 40, 128, -1)
        cv2.circle(img, (400, 400), 30, 80, -1)
        
        cv2.imwrite(os.path.join(test_img_dir, f"dummy_{i}.png"), img)
        
    print(f"   [SUCCESS] 2 dummy images created in '{test_img_dir}/'.")
    
    # 2. Execute generate_dataset
    print("\n2. Executing TargetTracker.generate_dataset()...")
    batch_size = 4
    num_samples = 8
    
    try:
        TargetTracker.generate_dataset(
            image_dir=test_img_dir,
            output_path=test_output_dir,
            batch_size=batch_size,
            num_of_samples=num_samples
        )
        print("   [SUCCESS] generate_dataset completed execution.")
    except Exception as e:
        print(f"   [FAILURE] generate_dataset failed: {e}")
        # Cleanup and raise
        shutil.rmtree(test_img_dir)
        shutil.rmtree(test_output_dir)
        raise e
        
    # 3. Verify Output Pickles
    print("\n3. Verifying output pickle files...")
    pickle_files = sorted(os.listdir(test_output_dir))
    expected_pickles = [f"dataset_{i}.pkl" for i in range(num_samples // batch_size)]
    
    assert pickle_files == expected_pickles, f"Expected pickles {expected_pickles}, but got {pickle_files}"
    print(f"   [SUCCESS] Created expected pickle files: {pickle_files}")
    
    # 4. Verify Shapes and Bounds of Loaded Data
    print("\n4. Loading and validating pickle data structure...")
    for pf in pickle_files:
        filepath = os.path.join(test_output_dir, pf)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            
        assert "inputs" in data, "Expected 'inputs' key in batch pickle data"
        assert "targets" in data, "Expected 'targets' key in batch pickle data"
        
        inputs = data["inputs"]
        targets = data["targets"]
        
        # Expected shapes:
        # Inputs: [hist_frames, hist_coords, prev_frames, prev_coords, curr_frames]
        assert len(inputs) == 5, f"Expected 5 inputs, got {len(inputs)}"
        
        hist_frames, hist_coords, prev_frames, prev_coords, curr_frames = inputs
        
        assert hist_frames.shape == (batch_size, 256, 256, 1), f"Expected shape {(batch_size, 256, 256, 1)}, got {hist_frames.shape}"
        assert prev_frames.shape == (batch_size, 256, 256, 1), f"Expected shape {(batch_size, 256, 256, 1)}, got {prev_frames.shape}"
        assert curr_frames.shape == (batch_size, 256, 256, 1), f"Expected shape {(batch_size, 256, 256, 1)}, got {curr_frames.shape}"
        
        assert hist_coords.shape == (batch_size, 2), f"Expected shape {(batch_size, 2)}, got {hist_coords.shape}"
        assert prev_coords.shape == (batch_size, 2), f"Expected shape {(batch_size, 2)}, got {prev_coords.shape}"
        assert targets.shape == (batch_size, 2), f"Expected shape {(batch_size, 2)}, got {targets.shape}"
        
        # Assert coordinate normalization bounds [0.0, 1.0]
        assert np.all(hist_coords >= 0.0) and np.all(hist_coords <= 1.0), "hist_coords contain values out of [0, 1]"
        assert np.all(prev_coords >= 0.0) and np.all(prev_coords <= 1.0), "prev_coords contain values out of [0, 1]"
        assert np.all(targets >= 0.0) and np.all(targets <= 1.0), "targets contain values out of [0, 1]"
        
        print(f"   [SUCCESS] Checked {pf}: All shapes and bounds are correct.")
        
    print("\n5. Cleaning up temporary test directories...")
    shutil.rmtree(test_img_dir)
    shutil.rmtree(test_output_dir)
    print("   [SUCCESS] Cleanup completed.")
    
    print("\n=== All Dataset Generator Tests Passed Successfully! ===")

if __name__ == "__main__":
    run_generator_tests()
