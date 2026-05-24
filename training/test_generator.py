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
    test_txt_list = "temp_test_images_list.txt"
    
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    print("1. Creating dummy test images...")
    # Create two dummy images of size 500x500 with simple shapes
    image_filepaths = []
    for i in range(2):
        img = np.zeros((500, 500), dtype=np.uint8)
        # Draw a square target in the center
        cv2.rectangle(img, (200, 200), (300, 300), 255, -1)
        # Draw some noise shapes
        cv2.circle(img, (100, 100), 40, 128, -1)
        cv2.circle(img, (400, 400), 30, 80, -1)
        
        path = os.path.abspath(os.path.join(test_img_dir, f"dummy_{i}.png"))
        cv2.imwrite(path, img)
        image_filepaths.append(path)
        
    print(f"   [SUCCESS] 2 dummy images created in '{test_img_dir}/'.")
    
    # Create a text file listing these images
    with open(test_txt_list, "w", encoding="utf-8") as f:
        for p in image_filepaths:
            f.write(p + "\n")
    print(f"   [SUCCESS] Image list file created: '{test_txt_list}'")
    
    # 2. Execute generate_dataset using Directory Mode
    print("\n2. Executing TargetTracker.generate_dataset() in Directory Mode...")
    batch_size = 4
    num_samples = 8
    
    try:
        TargetTracker.generate_dataset(
            images_path=test_img_dir,
            output_path=test_output_dir,
            batch_size=batch_size,
            num_of_samples=num_samples
        )
        print("   [SUCCESS] generate_dataset directory mode completed execution.")
    except Exception as e:
        print(f"   [FAILURE] generate_dataset directory mode failed: {e}")
        cleanup(test_img_dir, test_output_dir, test_txt_list)
        raise e
        
    # Verify Output Pickles
    verify_pickles(test_output_dir, batch_size, num_samples)
    
    # Clear directory mode output
    shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # 3. Execute generate_dataset using Text List Mode
    print("\n3. Executing TargetTracker.generate_dataset() in Text File List Mode...")
    try:
        TargetTracker.generate_dataset(
            images_path=test_txt_list,
            output_path=test_output_dir,
            batch_size=batch_size,
            num_of_samples=num_samples
        )
        print("   [SUCCESS] generate_dataset text file list mode completed execution.")
    except Exception as e:
        print(f"   [FAILURE] generate_dataset text file list mode failed: {e}")
        cleanup(test_img_dir, test_output_dir, test_txt_list)
        raise e
        
    # Verify Output Pickles from text file mode
    verify_pickles(test_output_dir, batch_size, num_samples)
    
    # 4. Final Cleanup
    print("\n4. Cleaning up temporary test directories...")
    cleanup(test_img_dir, test_output_dir, test_txt_list)
    print("   [SUCCESS] Cleanup completed.")
    
    print("\n=== All Dataset Generator Tests Passed Successfully! ===")

def verify_pickles(output_dir, batch_size, num_samples):
    pickle_files = sorted(os.listdir(output_dir))
    expected_pickles = [f"dataset_{i}.pkl" for i in range(num_samples // batch_size)]
    
    assert pickle_files == expected_pickles, f"Expected pickles {expected_pickles}, but got {pickle_files}"
    
    for pf in pickle_files:
        filepath = os.path.join(output_dir, pf)
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
        
    print(f"   [SUCCESS] Verified pickle outputs: All shapes and bounds are correct.")

def cleanup(img_dir, output_dir, txt_file):
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(txt_file):
        os.remove(txt_file)

if __name__ == "__main__":
    run_generator_tests()
