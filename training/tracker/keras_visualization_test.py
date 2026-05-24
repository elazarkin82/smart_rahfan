#!/usr/bin/env python3
import os
import sys
import argparse
import tkinter as tk

# Add parent directory to path to allow importing tracker module if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracker.dataset_visual_test import DatasetVisualizer

class KerasVisualizer(DatasetVisualizer):
    """
    Subclass of DatasetVisualizer that loads a trained Keras model,
    runs inference on generated dataset samples, and displays a fourth panel
    showing the model's predicted target coordinate alongside the ground truth.
    """
    def __init__(self, root, images_path, model_path):
        import tensorflow as tf
        
        # 1. Load model first so it is available during setup_ui and load_next_sample calls in super().__init__
        print(f"Loading Keras model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        
        # 2. Call parent class initializer
        super().__init__(root, images_path)
        
        # 3. Customize window properties
        self.root.title("Recursive Target Tracker - Keras Inference Visual Inspector")
        self.root.geometry("1140x420")  # Wider window to accommodate 4 frames side-by-side
        
    def setup_ui(self):
        # 1. Main Header
        header = tk.Label(
            self.root, 
            text="Recursive Target Tracker - Keras Model Inference Visualizer", 
            font=("Outfit", 16, "bold"), 
            bg="#121212", 
            fg="#ffffff"
        )
        header.pack(pady=15)
        
        # 2. Frames Container (Horizontal Layout)
        self.frames_frame = tk.Frame(self.root, bg="#121212")
        self.frames_frame.pack(pady=10)
        
        # 3. Individual Frame Slots (4 slots)
        self.hist_panel = self.create_frame_slot(self.frames_frame, "Historical Frame (Warped)")
        self.prev_panel = self.create_frame_slot(self.frames_frame, "Previous Frame (Warped)")
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
        self.status_bar.config(text="Generating next sample and running Keras model prediction...", fg="#ffcc00")
        self.root.update_idletasks()
        
        try:
            import pickle
            import numpy as np
            import tensorflow as tf
            from tracker.model import TargetTracker
            from tkinter import messagebox
            
            # Generate a single sample
            TargetTracker.generate_dataset(
                images_path=self.images_path,
                output_path=self.temp_dir,
                batch_size=1,
                num_of_samples=1
            )
            
            # Load the generated pickle
            pickle_path = os.path.join(self.temp_dir, "dataset_0.pkl")
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
                
            # Extract inputs and target
            inputs = data["inputs"]
            targets = data["targets"]
            
            hist_frames, hist_coords, prev_frames, prev_coords, curr_frames = inputs
            
            # Get the single sample
            hist_frame = hist_frames[0]
            hist_coord = hist_coords[0]
            prev_frame = prev_frames[0]
            prev_coord = prev_coords[0]
            curr_frame = curr_frames[0]
            curr_coord = targets[0] # expected output (ground truth)
            
            # Run model prediction
            # Convert inputs to float32 tensors
            hist_tensor = tf.convert_to_tensor(hist_frames, dtype=tf.float32)
            hist_coord_tensor = tf.convert_to_tensor(hist_coords, dtype=tf.float32)
            prev_tensor = tf.convert_to_tensor(prev_frames, dtype=tf.float32)
            prev_coord_tensor = tf.convert_to_tensor(prev_coords, dtype=tf.float32)
            curr_tensor = tf.convert_to_tensor(curr_frames, dtype=tf.float32)
            
            pred = self.model([hist_tensor, hist_coord_tensor, prev_tensor, prev_coord_tensor, curr_tensor], training=False)
            pred_coord = pred.numpy()[0]
            
            # Process and draw circles
            self.tk_img_hist = self.process_and_draw(hist_frame, hist_coord, "#ff3333")  # Red for hist
            self.tk_img_prev = self.process_and_draw(prev_frame, prev_coord, "#ff9900")  # Orange for prev
            self.tk_img_curr = self.process_and_draw(curr_frame, curr_coord, "#00e6ff")  # Cyan for ground truth
            self.tk_img_pred = self.process_and_draw(curr_frame, pred_coord, "#33ff33")  # Neon Green for predicted
            
            # Update GUI panels
            self.hist_panel.config(image=self.tk_img_hist)
            self.prev_panel.config(image=self.tk_img_prev)
            self.curr_panel.config(image=self.tk_img_curr)
            self.pred_panel.config(image=self.tk_img_pred)
            
            # Delete file to save space and prepare for next call
            if os.path.exists(pickle_path):
                os.remove(pickle_path)
                
            self.status_bar.config(
                text=f"Press [Space] for Next Sample  |  Press [Esc] to Exit  |  Source: {self.images_path}",
                fg="#aaaaaa"
            )
            
        except Exception as e:
            self.status_bar.config(text=f"Error during inference: {e}", fg="#ff3333")
            import traceback
            traceback.print_exc()
            from tkinter import messagebox
            messagebox.showerror("Dataset/Inference Error", f"Failed to generate sample or run inference:\n{e}")

def main():
    parser = argparse.ArgumentParser(description="TargetTracker Keras Model Inference Visualizer GUI")
    parser.add_argument(
        "--images_path", 
        required=True, 
        help="Path to directory containing raw images OR path to a .txt file listing image paths (one per line)"
    )
    parser.add_argument(
        "--model_path", 
        required=True, 
        help="Path to the Keras model file (.keras) to load for inference"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.images_path):
        print(f"Error: Images path '{args.images_path}' does not exist.")
        sys.exit(1)
        
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist.")
        sys.exit(1)
        
    root = tk.Tk()
    app = KerasVisualizer(root, args.images_path, args.model_path)
    root.mainloop()

if __name__ == "__main__":
    main()
