#!/usr/bin/env python3
import os
import sys
import argparse
import tkinter as tk
import pickle
import numpy as np
import cv2
import shutil
from PIL import Image, ImageDraw, ImageTk

# Add parent directory to path to allow importing tracker module if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracker.dataset_visual_test import DatasetVisualizer

class Tracker2DatasetVisualizer(DatasetVisualizer):
    """
    Tkinter Visual Dataset Inspector tailored specifically for the FCN model (tracker_model2).
    It displays 4 panels:
      1. Historical Frame (Circular masked at r=128, warped, showing red debug target circle).
      2. Previous Frame (Circular masked at r=50, warped, showing orange debug target circle).
      3. Current Frame (Unmasked search region, showing cyan debug target circle).
      4. Target Heatmap (Resized 64x64 Gaussian heatmap rendered with a glowing JET colormap).
    """
    def __init__(self, root, images_path):
        # 1. Call parent class initializer
        super().__init__(root, images_path)
        
        # 2. Customize window properties
        self.root.title("Recursive Target Tracker - FCN Heatmap Dataset Inspector")
        self.root.geometry("1140x420")  # Wider window to accommodate 4 frames side-by-side
        
    def setup_ui(self):
        # 1. Main Header
        header = tk.Label(
            self.root, 
            text="Recursive Target Tracker - FCN Masked Heatmap Dataset Visualizer", 
            font=("Outfit", 16, "bold"), 
            bg="#121212", 
            fg="#ffffff"
        )
        header.pack(pady=15)
        
        # 2. Frames Container (Horizontal Layout)
        self.frames_frame = tk.Frame(self.root, bg="#121212")
        self.frames_frame.pack(pady=10)
        
        # 3. Individual Frame Slots (4 slots)
        self.hist_panel = self.create_frame_slot(self.frames_frame, "Historical Frame (Masked)")
        self.prev_panel = self.create_frame_slot(self.frames_frame, "Previous Frame (Masked)")
        self.curr_panel = self.create_frame_slot(self.frames_frame, "Current Frame (Unmasked)")
        self.heatmap_panel = self.create_frame_slot(self.frames_frame, "Target Heatmap (64x64 Gaussian)")
        
        # 4. Footer Status Bar
        self.status_bar = tk.Label(
            self.root, 
            text=f"Press [Space] for Next Sample  |  Press [Esc] to Exit  |  Source: {self.images_path}",
            font=("Inter", 10),
            bg="#1c1c1c",
            fg="#aaaaaa",
            anchor="w",
            padx=15,
            pady=8
        )
        self.status_bar.pack(side="bottom", fill="x")
        
    def load_next_sample(self):
        self.status_bar.config(text="Generating next masked sample via TargetTracker2...", fg="#ffcc00")
        self.root.update_idletasks()
        
        try:
            from tracker.tracker_model2 import TargetTracker2
            from tkinter import messagebox
            
            # Generate a single masked sample
            TargetTracker2.generate_dataset(
                images_path=self.images_path,
                output_path=self.temp_dir,
                batch_size=1,
                num_of_samples=1
            )
            
            # Load the generated pickle
            pickle_path = os.path.join(self.temp_dir, "dataset_0.pkl")
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
                
            # Extract inputs, targets, and debug coordinates
            inputs = data["inputs"]
            targets = data["targets"]
            debug_coords = data["debug_coords"]
            
            hist_frames, prev_frames, curr_frames = inputs
            hist_frame = hist_frames[0]
            prev_frame = prev_frames[0]
            curr_frame = curr_frames[0]
            
            # Get 64x64 target heatmap
            target_heatmap = targets[0] # shape: (64, 64, 1)
            
            # Get debug target coordinates
            hist_coord = debug_coords["hist"][0]
            prev_coord = debug_coords["prev"][0]
            curr_coord = debug_coords["curr"][0]
            
            # Process and draw circles on the first 3 frames
            self.tk_img_hist = self.process_and_draw(hist_frame, hist_coord, "#ff3333")  # Red for historical
            self.tk_img_prev = self.process_and_draw(prev_frame, prev_coord, "#ff9900")  # Orange for previous
            self.tk_img_curr = self.process_and_draw(curr_frame, curr_coord, "#00e6ff")  # Cyan for Ground Truth
            
            # Process and draw the 64x64 Gaussian heatmap in JET colormap
            # 1. Resize heatmap to 256x256
            heatmap_resized = cv2.resize(target_heatmap[:, :, 0], (256, 256), interpolation=cv2.INTER_CUBIC)
            # 2. Scale to 0-255 uint8
            heatmap_uint8 = (np.clip(heatmap_resized, 0.0, 1.0) * 255.0).astype(np.uint8)
            # 3. Apply glowing JET colormap (blue = cold, red = hot peak)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            # 4. Convert BGR to RGB for PIL
            heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            # 5. Convert to ImageTk
            pil_heatmap = Image.fromarray(heatmap_color_rgb)
            self.tk_img_heatmap = ImageTk.PhotoImage(pil_heatmap)
            
            # Update GUI panels
            self.hist_panel.config(image=self.tk_img_hist)
            self.prev_panel.config(image=self.tk_img_prev)
            self.curr_panel.config(image=self.tk_img_curr)
            self.heatmap_panel.config(image=self.tk_img_heatmap)
            
            # Delete file to prepare for next step
            if os.path.exists(pickle_path):
                os.remove(pickle_path)
                
            self.status_bar.config(
                text=f"Press [Space] for Next Sample  |  Press [Esc] to Exit  |  Source: {self.images_path}",
                fg="#aaaaaa"
            )
            
        except Exception as e:
            self.status_bar.config(text=f"Error generating sample: {e}", fg="#ff3333")
            import traceback
            traceback.print_exc()
            from tkinter import messagebox
            messagebox.showerror("Dataset Error", f"Failed to generate and visualize sample:\n{e}")

def main():
    parser = argparse.ArgumentParser(description="TargetTracker2 Masked Heatmap Dataset Visualizer GUI")
    parser.add_argument(
        "--images_path", 
        required=True, 
        help="Path to directory containing raw images OR path to a .txt file listing image paths (one per line)"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.images_path):
        print(f"Error: Images path '{args.images_path}' does not exist.")
        sys.exit(1)
        
    root = tk.Tk()
    app = Tracker2DatasetVisualizer(root, args.images_path)
    root.mainloop()

if __name__ == "__main__":
    main()
