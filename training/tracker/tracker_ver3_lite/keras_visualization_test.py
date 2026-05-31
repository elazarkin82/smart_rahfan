#!/usr/bin/env python3
import os
import sys
import argparse
import tkinter as tk
import pickle
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageTk

class KerasFCNVisualizer:
    """
    Tkinter Visual Inspector adapted for TargetTracker3-Lite (2-Frame Symmetric Model).
    Loads pre-generated 2-frame pickles, runs live inference, computes sub-pixel coordinates
    using Center of Mass, and renders expected/predicted spatial heatmaps.
    """
    def __init__(self, root, dataset_dir, model_path):
        import tensorflow as tf
        
        # 1. Load the Keras model
        print(f"Loading Keras FCN model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        
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
        
        # 4. Configure window properties
        self.root = root
        self.root.title("TargetTracker3-Lite - Keras FCN Heatmap Visual Inspector")
        self.root.geometry("860x900")  # Resized neatly for 3 panels side-by-side
        self.root.configure(bg="#121212")
        self.root.resizable(False, False)
        
        # 5. Window closing handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 6. Build UI layout
        self.setup_ui()
        
        # 7. Bind keyboard events
        self.root.bind("<space>", lambda e: self.load_next_sample())
        self.root.bind("<Escape>", lambda e: self.on_close())
        
        # 8. Load first sample
        self.load_next_sample()

    def process_and_draw(self, frame_np, coords_np, circle_color):
        """
        Draws outer glowing circle and center dot at target coords and converts to PhotoImage.
        """
        # Extract Channel 0 (Grayscale Image) from frame_np (can be 2-channel or 1-channel)
        img_8u = (frame_np[:, :, 0] * 255.0).astype(np.uint8)
        
        # Convert to RGB to allow drawing a colored circle
        img_rgb = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Draw target markers
        draw = ImageDraw.Draw(pil_img)
        r = 6  # Circle radius
        x_px = int(coords_np[0] * 256.0)
        y_px = int(coords_np[1] * 256.0)
        
        draw.ellipse([x_px - r, y_px - r, x_px + r, y_px + r], outline=circle_color, width=2)
        draw.ellipse([x_px - 2, y_px - 2, x_px + 2, y_px + 2], fill=circle_color)
        
        return ImageTk.PhotoImage(pil_img)
        
    def create_frame_slot_with_coord(self, parent, label_text):
        container = tk.Frame(parent, bg="#121212", padx=10)
        container.pack(side="left")
        
        # Title above image
        lbl = tk.Label(
            container, 
            text=label_text, 
            font=("Inter", 11, "bold"), 
            bg="#121212", 
            fg="#00e6ff" if "Ground Truth" in label_text else ("#33ff33" if "Model Prediction" in label_text else "#e0e0e0")
        )
        lbl.pack(pady=5)
        
        # Canvas/Panel for image
        panel = tk.Label(container, bg="#1a1a1a", bd=1, relief="solid", width=256, height=256)
        panel.pack()
        
        # Coordinate label below image
        coord_lbl = tk.Label(
            container, 
            text="Norm: [N/A, N/A]\nPixel: [N/A, N/A]", 
            font=("Consolas", 10), 
            bg="#121212", 
            fg="#888888",
            justify="center",
            pady=5
        )
        coord_lbl.pack()
        
        return panel, coord_lbl

    def create_heatmap_slot(self, parent, label_text, title_color):
        container = tk.Frame(parent, bg="#121212", padx=10)
        container.pack(side="left")
        
        lbl = tk.Label(
            container, 
            text=label_text, 
            font=("Inter", 11, "bold"), 
            bg="#121212", 
            fg=title_color
        )
        lbl.pack(pady=5)
        
        panel = tk.Label(container, bg="#1a1a1a", bd=1, relief="solid", width=256, height=256)
        panel.pack()
        return panel

    def setup_ui(self):
        # 1. Main Header
        header = tk.Label(
            self.root, 
            text="TargetTracker3-Lite - Keras FCN Heatmap Visualizer", 
            font=("Outfit", 15, "bold"), 
            bg="#121212", 
            fg="#ffffff"
        )
        header.pack(pady=15)
        
        # 2. Frames Container (Row 1: Images with Coordinate Markers)
        self.frames_frame = tk.Frame(self.root, bg="#121212")
        self.frames_frame.pack(pady=10)
        
        # 3. 3 slots: Historical (Image+Mask), Current Ground Truth, Current Model Prediction
        self.hist_panel, self.hist_coord_lbl = self.create_frame_slot_with_coord(self.frames_frame, "Historical Frame (Channel 0)")
        self.curr_panel, self.curr_coord_lbl = self.create_frame_slot_with_coord(self.frames_frame, "Current Frame (Ground Truth)")
        self.pred_panel, self.pred_coord_lbl = self.create_frame_slot_with_coord(self.frames_frame, "Current Frame (Model Prediction)")
        
        # 4. Heatmap Frames Container (Row 2: Spatial Heatmaps)
        self.heatmap_frames_frame = tk.Frame(self.root, bg="#121212")
        self.heatmap_frames_frame.pack(pady=15)
        
        # 5. 3 slots: Expected (GT), Predicted (Model), Heatmap Overlay
        self.exp_heatmap_panel = self.create_heatmap_slot(self.heatmap_frames_frame, "Expected Heatmap (GT)", "#ff3333")
        self.pred_heatmap_panel = self.create_heatmap_slot(self.heatmap_frames_frame, "Predicted Heatmap (Model)", "#33ff33")
        self.overlay_heatmap_panel = self.create_heatmap_slot(self.heatmap_frames_frame, "Heatmap Overlay (Overlap)", "#ffff33")
        
        # 6. Footer Status Bar
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
        self.status_bar.config(text="Loading dataset sample and running FCN inference...", fg="#ffcc00")
        self.root.update_idletasks()
        
        try:
            import tensorflow as tf
            
            # Load new batch if needed
            if self.current_batch_data is None or self.current_sample_idx >= self.current_batch_size - 1:
                if self.current_batch_data is not None:
                    self.current_pickle_idx = (self.current_pickle_idx + 1) % len(self.pickle_files)
                
                pickle_name = self.pickle_files[self.current_pickle_idx]
                pickle_path = os.path.join(self.dataset_dir, pickle_name)
                
                print(f"Loading batch pickle file: {pickle_path}...")
                with open(pickle_path, "rb") as f:
                    self.current_batch_data = pickle.load(f)
                
                if "inputs" not in self.current_batch_data or "debug_coords" not in self.current_batch_data:
                    raise ValueError(f"Pickle {pickle_name} is missing 'inputs' or 'debug_coords' keys.")
                
                self.current_sample_idx = 0
                self.current_batch_size = len(self.current_batch_data["inputs"][0])
            else:
                self.current_sample_idx += 1
                
            inputs = self.current_batch_data["inputs"]
            debug_coords = self.current_batch_data["debug_coords"]
            
            hist_frames, curr_frames = inputs
            hist_frame = hist_frames[self.current_sample_idx]  # shape: (256, 256, 2)
            curr_frame = curr_frames[self.current_sample_idx]  # shape: (256, 256, 1)
            
            hist_coord = debug_coords["hist"][self.current_sample_idx]
            curr_coord = debug_coords["curr"][self.current_sample_idx]  # Ground truth target
            
            # 1. Convert to tensors and expand batch dimension
            hist_tensor = tf.convert_to_tensor(np.expand_dims(hist_frame, axis=0), dtype=tf.float32)
            curr_tensor = tf.convert_to_tensor(np.expand_dims(curr_frame, axis=0), dtype=tf.float32)
            
            # 2. Run model forward pass to obtain predicted 64x64 heatmap
            pred = self.model([hist_tensor, curr_tensor], training=False)
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
                
            # Process and draw markers
            self.tk_img_hist = self.process_and_draw(hist_frame, hist_coord, "#ff3333")  # Red for historical
            self.tk_img_curr = self.process_and_draw(curr_frame, curr_coord, "#00e6ff")  # Cyan for Ground Truth
            self.tk_img_pred = self.process_and_draw(curr_frame, pred_coord, "#33ff33")  # Neon Green for Prediction
            
            # Update GUI panels
            self.hist_panel.config(image=self.tk_img_hist)
            self.curr_panel.config(image=self.tk_img_curr)
            self.pred_panel.config(image=self.tk_img_pred)
            
            # Update Coordinate Labels
            self.hist_coord_lbl.config(
                text=f"Norm: [{hist_coord[0]:.4f}, {hist_coord[1]:.4f}]\nPixel: [{int(hist_coord[0] * 256.0)}, {int(hist_coord[1] * 256.0)}]",
                fg="#ff5555"
            )
            self.curr_coord_lbl.config(
                text=f"Norm: [{curr_coord[0]:.4f}, {curr_coord[1]:.4f}]\nPixel: [{int(curr_coord[0] * 256.0)}, {int(curr_coord[1] * 256.0)}]",
                fg="#00e6ff"
            )
            
            # Compute tracking distance error in pixels (scaled to 256x256 frame)
            error_px = np.sqrt((pred_coord[0] - curr_coord[0])**2 + (pred_coord[1] - curr_coord[1])**2) * 256.0
            self.pred_coord_lbl.config(
                text=f"Norm: [{pred_coord[0]:.4f}, {pred_coord[1]:.4f}]\nPixel: [{int(pred_coord[0] * 256.0)}, {int(pred_coord[1] * 256.0)}]",
                fg="#33ff33"
            )
            
            # --- ROW 2: HEATMAP PROCESS & RENDER ---
            # 1. Retrieve the target heatmap from batch targets
            target_heatmap = self.current_batch_data["targets"][self.current_sample_idx]  # shape: (64, 64, 1)
            
            # 2. Extract and scale Expected (GT) and Predicted (Model) matrices to [0, 255]
            h_gt = (target_heatmap[:, :, 0] * 255.0).astype(np.uint8)
            h_pred = (pred_heatmap[:, :, 0] * 255.0).astype(np.uint8)
            
            # 3. Resize to 256x256 using bilinear interpolation for smooth gradients
            h_gt_256 = cv2.resize(h_gt, (256, 256), interpolation=cv2.INTER_LINEAR)
            h_pred_256 = cv2.resize(h_pred, (256, 256), interpolation=cv2.INTER_LINEAR)
            
            # 4. Construct RGB glow arrays:
            # Expected (Red Glow): R = GT, G = 0, B = 0
            exp_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            exp_rgb[:, :, 0] = h_gt_256
            
            # Predicted (Green Glow): R = 0, G = Model, B = 0
            pred_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            pred_rgb[:, :, 1] = h_pred_256
            
            # Overlay (Yellow Overlap): R = GT, G = Model, B = 0
            overlay_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            overlay_rgb[:, :, 0] = h_gt_256
            overlay_rgb[:, :, 1] = h_pred_256
            
            # 5. Convert to PhotoImage structures
            self.tk_img_exp_hm = ImageTk.PhotoImage(Image.fromarray(exp_rgb))
            self.tk_img_pred_hm = ImageTk.PhotoImage(Image.fromarray(pred_rgb))
            self.tk_img_overlay_hm = ImageTk.PhotoImage(Image.fromarray(overlay_rgb))
            
            # 6. Apply images to GUI labels
            self.exp_heatmap_panel.config(image=self.tk_img_exp_hm)
            self.pred_heatmap_panel.config(image=self.tk_img_pred_hm)
            self.overlay_heatmap_panel.config(image=self.tk_img_overlay_hm)
            
            pickle_name = self.pickle_files[self.current_pickle_idx]
            self.status_bar.config(
                text=f"Batch: {pickle_name} | Sample: {self.current_sample_idx + 1}/{self.current_batch_size} | L2 Error: {error_px:.2f} px | Press [Space] for Next",
                fg="#aaaaaa"
            )
            
        except Exception as e:
            self.status_bar.config(text=f"Error during FCN inference: {e}", fg="#ff3333")
            import traceback
            traceback.print_exc()

    def on_close(self):
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser(description="TargetTracker3-Lite Keras FCN Heatmap Visualizer GUI")
    parser.add_argument(
        "--dataset_dir", 
        default="dataset_generation_project/video_dataset/",
        help="Path to directory containing the .pkl batch files"
    )
    parser.add_argument(
        "--model_path", 
        required=True, 
        help="Path to the trained Keras model (.keras) to load for inference"
    )
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = args.dataset_dir
    if not os.path.isabs(dataset_dir) and not os.path.exists(dataset_dir):
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
    app = KerasFCNVisualizer(root, dataset_dir, args.model_path)
    root.mainloop()

if __name__ == "__main__":
    main()
