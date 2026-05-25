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

class KerasFCNVisualizer(DatasetVisualizer):
    """
    Tkinter Visual Inspector tailored for the Fully Convolutional tracker (TargetTracker2).
    It processes masked input frames, runs live inference to predict a 64x64 heatmap,
    extracts the sub-pixel coordinate using a CPU-based Center of Mass calculation,
    and overlays target markers dynamically across a 4-panel GUI layout.
    """
    def __init__(self, root, images_path, model_path):
        import tensorflow as tf
        
        # 1. Load the uncompiled FCN model
        print(f"Loading Keras FCN model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        
        # 2. Call parent class initializer
        super().__init__(root, images_path)
        
        # 3. Customize window properties
        self.root.title("Recursive Target Tracker - Keras FCN Heatmap Visual Inspector")
        self.root.geometry("1140x420")  # Accommodate 4 horizontal panels
        
    def setup_ui(self):
        # 1. Main Header
        header = tk.Label(
            self.root, 
            text="Recursive Target Tracker - Keras FCN Heatmap Inference Visualizer", 
            font=("Outfit", 16, "bold"), 
            bg="#121212", 
            fg="#ffffff"
        )
        header.pack(pady=15)
        
        # 2. Frames Container (Horizontal Layout)
        self.frames_frame = tk.Frame(self.root, bg="#121212")
        self.frames_frame.pack(pady=10)
        
        # 3. Individual Frame Slots (4 slots)
        self.hist_panel = self.create_frame_slot(self.frames_frame, "Historical Frame (Masked/Warped)")
        self.prev_panel = self.create_frame_slot(self.frames_frame, "Previous Frame (Masked/Warped)")
        self.curr_panel = self.create_frame_slot(self.frames_frame, "Current Frame (Ground Truth)")
        self.pred_panel = self.create_frame_slot(self.frames_frame, "Current Frame (Model Prediction)")
        
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
        self.status_bar.config(text="Generating next masked sample and running FCN inference...", fg="#ffcc00")
        self.root.update_idletasks()
        
        try:
            import tensorflow as tf
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
            
            hist_coord = debug_coords["hist"][0]
            prev_coord = debug_coords["prev"][0]
            curr_coord = debug_coords["curr"][0] # ground truth coordinate
            
            # 1. Convert inputs to tensors
            hist_tensor = tf.convert_to_tensor(hist_frames, dtype=tf.float32)
            prev_tensor = tf.convert_to_tensor(prev_frames, dtype=tf.float32)
            curr_tensor = tf.convert_to_tensor(curr_frames, dtype=tf.float32)
            
            # 2. Run model forward pass to obtain predicted 64x64 heatmap
            pred = self.model([hist_tensor, prev_tensor, curr_tensor], training=False)
            pred_heatmap = pred.numpy()[0]  # shape: (64, 64, 1)
            
            # 3. CPU-based Center of Mass (weighted average) calculation for sub-pixel accuracy
            heatmap = pred_heatmap[:, :, 0]
            # Threshold to focus on activation peak region and eliminate background noise
            threshold = 0.1
            heatmap_thresh = np.where(heatmap > threshold, heatmap, 0.0)
            
            total_mass = np.sum(heatmap_thresh)
            if total_mass > 1e-6:
                x_indices = np.arange(64)
                y_indices = np.arange(64)
                x_grid, y_grid = np.meshgrid(x_indices, y_indices)
                
                # Compute weighted average
                x_c = np.sum(x_grid * heatmap_thresh) / total_mass
                y_c = np.sum(y_grid * heatmap_thresh) / total_mass
                
                # Normalize coordinates to [0, 1]
                pred_coord = [x_c / 64.0, y_c / 64.0]
            else:
                pred_coord = [0.5, 0.5] # Fallback if no activation
                
            # Process and draw markers
            self.tk_img_hist = self.process_and_draw(hist_frame, hist_coord, "#ff3333")  # Red for historical
            self.tk_img_prev = self.process_and_draw(prev_frame, prev_coord, "#ff9900")  # Orange for previous
            self.tk_img_curr = self.process_and_draw(curr_frame, curr_coord, "#00e6ff")  # Cyan for Ground Truth
            self.tk_img_pred = self.process_and_draw(curr_frame, pred_coord, "#33ff33")  # Neon Green for Prediction
            
            # Update GUI panels
            self.hist_panel.config(image=self.tk_img_hist)
            self.prev_panel.config(image=self.tk_img_prev)
            self.curr_panel.config(image=self.tk_img_curr)
            self.pred_panel.config(image=self.tk_img_pred)
            
            # Delete file to prepare for next step
            if os.path.exists(pickle_path):
                os.remove(pickle_path)
                
            self.status_bar.config(
                text=f"Press [Space] for Next Sample  |  Press [Esc] to Exit  |  Source: {self.images_path}",
                fg="#aaaaaa"
            )
            
        except Exception as e:
            self.status_bar.config(text=f"Error during FCN inference: {e}", fg="#ff3333")
            import traceback
            traceback.print_exc()
            from tkinter import messagebox
            messagebox.showerror("Inference Error", f"Failed to run FCN inference:\n{e}")

def main():
    parser = argparse.ArgumentParser(description="TargetTracker2 Keras FCN Heatmap Visualizer GUI")
    parser.add_argument(
        "--images_path", 
        required=True, 
        help="Path to directory containing raw images OR path to a .txt file listing image paths (one per line)"
    )
    parser.add_argument(
        "--model_path", 
        required=True, 
        help="Path to the trained tracker2 Keras model (.keras) to load for inference"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.images_path):
        print(f"Error: Images path '{args.images_path}' does not exist.")
        sys.exit(1)
        
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist.")
        sys.exit(1)
        
    root = tk.Tk()
    app = KerasFCNVisualizer(root, args.images_path, args.model_path)
    root.mainloop()

if __name__ == "__main__":
    main()
