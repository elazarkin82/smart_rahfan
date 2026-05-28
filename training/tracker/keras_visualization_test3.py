#!/usr/bin/env python3
import os
import sys
import argparse
import tkinter as tk
import pickle
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageTk

# Add parent directory to path to allow importing tracker module if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracker.dataset_visual_test import DatasetVisualizer

class KerasFCNVisualizer3(DatasetVisualizer):
    """
    Tkinter Visual Inspector tailored for TargetTracker3.
    Loads pre-generated 2-channel pickles, runs live inference to predict a 64x64 heatmap,
    extracts the sub-pixel coordinate using a CPU-based Center of Mass calculation,
    and overlays target markers dynamically across a 4-panel GUI layout.
    """
    def __init__(self, root, dataset_dir, model_path):
        import tensorflow as tf
        
        # 1. Load the Keras model
        print(f"Loading Keras FCN model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        
        # 2. Discover all pre-generated pickle batches
        self.dataset_dir = dataset_dir
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{self.dataset_dir}' does not exist.")
            
        self.pickle_files = []
        for f in os.listdir(self.dataset_dir):
            file_path = os.path.join(self.dataset_dir, f)
            if os.path.isfile(file_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in ('.pkl', '.pickle'):
                    self.pickle_files.append(f)
                    
        if not self.pickle_files:
            raise FileNotFoundError(f"No pickle files (.pkl/.pickle) found in '{self.dataset_dir}'.")
            
        self.pickle_files.sort()
        print(f"Discovered {len(self.pickle_files)} dataset pickle files.")
        
        # 3. Track sample iteration indices
        self.current_pickle_idx = 0
        self.current_sample_idx = -1
        self.current_batch_data = None
        self.current_batch_size = 0
        
        # 4. Call parent class initializer
        super().__init__(root, dataset_dir)
        
        # 5. Customize window properties
        self.root.title("Recursive Target Tracker - Keras FCN 3 Heatmap Visual Inspector")
        self.root.geometry("1140x420")  # Accommodate 4 horizontal panels
        
    def setup_ui(self):
        # 1. Main Header
        header = tk.Label(
            self.root, 
            text="Recursive Target Tracker - Keras FCN 3 Heatmap Inference Visualizer", 
            font=("Outfit", 16, "bold"), 
            bg="#121212", 
            fg="#ffffff"
        )
        header.pack(pady=15)
        
        # 2. Frames Container (Horizontal Layout)
        self.frames_frame = tk.Frame(self.root, bg="#121212")
        self.frames_frame.pack(pady=10)
        
        # 3. Individual Frame Slots (4 slots)
        self.hist_panel = self.create_frame_slot(self.frames_frame, "Historical Frame (Channel 0)")
        self.prev_panel = self.create_frame_slot(self.frames_frame, "Previous Frame (Channel 0)")
        self.curr_panel = self.create_frame_slot(self.frames_frame, "Current Frame (Ground Truth)")
        self.pred_panel = self.create_frame_slot(self.frames_frame, "Current Frame (Model Prediction)")
        
        # 4. Footer Status Bar
        self.status_bar = tk.Label(
            self.root, 
            text=f"Press [Space] for Next Sample  |  Press [Esc] to Exit  |  Dataset: {self.dataset_dir}",
            font=("Inter", 10),
            bg="#1c1c1c",
            fg="#aaaaaa",
            anchor="w",
            padx=15,
            pady=8
        )
        self.status_bar.pack(side="bottom", fill="x")
        
    def load_next_sample(self):
        self.status_bar.config(text="Loading dataset sample and running FCN 3 inference...", fg="#ffcc00")
        self.root.update_idletasks()
        
        try:
            import tensorflow as tf
            from tkinter import messagebox
            
            # Load new batch if needed
            if self.current_batch_data is None or self.current_sample_idx >= self.current_batch_size - 1:
                if self.current_batch_data is not None:
                    # Move to next pickle file
                    self.current_pickle_idx = (self.current_pickle_idx + 1) % len(self.pickle_files)
                
                pickle_name = self.pickle_files[self.current_pickle_idx]
                pickle_path = os.path.join(self.dataset_dir, pickle_name)
                
                print(f"Loading batch pickle file: {pickle_path}...")
                with open(pickle_path, "rb") as f:
                    self.current_batch_data = pickle.load(f)
                
                # Verify batch keys
                if "inputs" not in self.current_batch_data or "debug_coords" not in self.current_batch_data:
                    raise ValueError(f"Pickle {pickle_name} is missing 'inputs' or 'debug_coords' keys.")
                
                self.current_sample_idx = 0
                self.current_batch_size = len(self.current_batch_data["inputs"][0])
            else:
                self.current_sample_idx += 1
                
            # Extract inputs, targets, and coordinates for current index
            inputs = self.current_batch_data["inputs"]
            debug_coords = self.current_batch_data["debug_coords"]
            
            hist_frames, prev_frames, curr_frames = inputs
            hist_frame = hist_frames[self.current_sample_idx]
            prev_frame = prev_frames[self.current_sample_idx]
            curr_frame = curr_frames[self.current_sample_idx]
            
            hist_coord = debug_coords["hist"][self.current_sample_idx]
            prev_coord = debug_coords["prev"][self.current_sample_idx]
            curr_coord = debug_coords["curr"][self.current_sample_idx]  # Ground truth target
            
            # 1. Convert to tensors and expand batch dimension
            hist_tensor = tf.convert_to_tensor(np.expand_dims(hist_frame, axis=0), dtype=tf.float32)
            prev_tensor = tf.convert_to_tensor(np.expand_dims(prev_frame, axis=0), dtype=tf.float32)
            curr_tensor = tf.convert_to_tensor(np.expand_dims(curr_frame, axis=0), dtype=tf.float32)
            
            # 2. Run model forward pass to obtain predicted 64x64 heatmap
            pred = self.model([hist_tensor, prev_tensor, curr_tensor], training=False)
            pred_heatmap = pred.numpy()[0]  # shape: (64, 64, 1)
            
            # 3. CPU-based Center of Mass calculation for sub-pixel accuracy
            heatmap = pred_heatmap[:, :, 0]
            threshold = 0.1
            heatmap_thresh = np.where(heatmap > threshold, heatmap, 0.0)
            
            total_mass = np.sum(heatmap_thresh)
            if total_mass > 1e-6:
                x_indices = np.arange(64)
                y_indices = np.arange(64)
                x_grid, y_grid = np.meshgrid(x_indices, y_indices)
                
                x_c = np.sum(x_grid * heatmap_thresh) / total_mass
                y_c = np.sum(y_grid * heatmap_thresh) / total_mass
                
                pred_coord = [x_c / 64.0, y_c / 64.0]
            else:
                pred_coord = [0.5, 0.5]  # Fallback if no activation
                
            # Process and draw markers (inherited process_and_draw extracts Channel 0 automatically)
            self.tk_img_hist = self.process_and_draw(hist_frame, hist_coord, "#ff3333")  # Red for historical
            self.tk_img_prev = self.process_and_draw(prev_frame, prev_coord, "#ff9900")  # Orange for previous
            self.tk_img_curr = self.process_and_draw(curr_frame, curr_coord, "#00e6ff")  # Cyan for Ground Truth
            self.tk_img_pred = self.process_and_draw(curr_frame, pred_coord, "#33ff33")  # Neon Green for Prediction
            
            # Update GUI panels
            self.hist_panel.config(image=self.tk_img_hist)
            self.prev_panel.config(image=self.tk_img_prev)
            self.curr_panel.config(image=self.tk_img_curr)
            self.pred_panel.config(image=self.tk_img_pred)
            
            pickle_name = self.pickle_files[self.current_pickle_idx]
            self.status_bar.config(
                text=f"Batch: {pickle_name} | Sample: {self.current_sample_idx + 1}/{self.current_batch_size}  |  Press [Space] for Next  |  Press [Esc] to Exit",
                fg="#aaaaaa"
            )
            
        except Exception as e:
            self.status_bar.config(text=f"Error during FCN inference: {e}", fg="#ff3333")
            import traceback
            traceback.print_exc()
            from tkinter import messagebox
            messagebox.showerror("Inference Error", f"Failed to run FCN inference:\n{e}")

    def on_close(self):
        # Simply destroy root, no temp dir cleanup needed
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser(description="TargetTracker3 Keras FCN Heatmap Visualizer GUI")
    parser.add_argument(
        "--dataset_dir", 
        default="dataset_generation_project/video_dataset/",
        help="Path to directory containing the .pkl batch files (default: dataset_generation_project/video_dataset/)"
    )
    parser.add_argument(
        "--model_path", 
        required=True, 
        help="Path to the trained tracker3 Keras model (.keras) to load for inference"
    )
    args = parser.parse_args()
    
    # Resolve relative paths relative to script's directory if run from there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = args.dataset_dir
    if not os.path.isabs(dataset_dir) and not os.path.exists(dataset_dir):
        # Try relative to training/tracker/
        alt_path = os.path.join(script_dir, dataset_dir)
        if os.path.exists(alt_path):
            dataset_dir = alt_path
            
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{args.dataset_dir}' does not exist.")
        sys.exit(1)
        
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist.")
        sys.exit(1)
        
    root = tk.Tk()
    app = KerasFCNVisualizer3(root, dataset_dir, args.model_path)
    root.mainloop()

if __name__ == "__main__":
    main()
