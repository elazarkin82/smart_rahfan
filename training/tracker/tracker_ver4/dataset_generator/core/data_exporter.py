import cv2
import os
import json
import time

class DataExporter:
    """
    Handles exporting RGB frames, drawing annotations, and saving metadata.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_capture(self, rgb_array, target_pixel=None, suffix="initial"):
        """
        Saves the RGB image. If target_pixel (x,y) is provided, draws a red circle.
        rgb_array is expected to be RGB. We convert to BGR for OpenCV saving.
        """
        timestamp = int(time.time() * 1000)
        # OpenCV requires BGR
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        
        # Draw target visualization if provided
        if target_pixel is not None:
            x, y = target_pixel
            # Draw empty circle
            cv2.circle(bgr_array, (int(x), int(y)), 20, (0, 0, 255), 2)
            # Draw solid center dot
            cv2.circle(bgr_array, (int(x), int(y)), 3, (0, 0, 255), -1)
            
        filename = f"{timestamp}_{suffix}.png"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, bgr_array)
        print(f"[DataExporter] Saved {filename}")
        return filepath
