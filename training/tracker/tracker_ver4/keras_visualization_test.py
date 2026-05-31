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
    def __init__(self, root, dataset_dir, model_path):
        import tensorflow as tf
        
        print(f"Loading Keras TargetTrackerVer4 from {model_path}...")
        self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        
        self.dataset_dir = dataset_dir
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{self.dataset_dir}' does not exist.")
            
        self.pickle_files = [f for f in os.listdir(self.dataset_dir) if f.startswith('train_') and f.endswith('.pkl')]
        if not self.pickle_files:
            raise FileNotFoundError(f"No pickle files found in '{self.dataset_dir}'.")
            
        self.pickle_files.sort()
        print(f"Discovered {len(self.pickle_files)} dataset pickle files.")
        
        self.current_pickle_idx = 0
        self.current_sample_idx = -1
        self.current_batch_data = None
        self.current_batch_size = 0
        
        self.root = root
        self.root.title("TargetTrackerVer4 - Live Inference Visualizer")
        self.root.geometry("860x500")
        self.root.configure(bg="#121212")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.setup_ui()
        
        self.root.bind("<space>", lambda e: self.load_next_sample())
        self.root.bind("<Escape>", lambda e: self.on_close())
        
        self.load_next_sample()

    def process_and_draw(self, img_256, coords_norm, circle_color):
        pil_img = Image.fromarray(img_256)
        draw = ImageDraw.Draw(pil_img)
        r = 6
        x_px = int(coords_norm[0] * 256.0)
        y_px = int(coords_norm[1] * 256.0)
        
        draw.ellipse([x_px - r, y_px - r, x_px + r, y_px + r], outline=circle_color, width=2)
        draw.ellipse([x_px - 2, y_px - 2, x_px + 2, y_px + 2], fill=circle_color)
        
        return ImageTk.PhotoImage(pil_img)
        
    def create_frame_slot(self, parent, label_text):
        container = tk.Frame(parent, bg="#121212", padx=10)
        container.pack(side="left")
        
        lbl = tk.Label(container, text=label_text, font=("Inter", 11, "bold"), bg="#121212", fg="#e0e0e0")
        lbl.pack(pady=5)
        
        panel = tk.Label(container, bg="#1a1a1a", bd=1, relief="solid", width=256, height=256)
        panel.pack()
        
        coord_lbl = tk.Label(container, text="[N/A]", font=("Consolas", 10), bg="#121212", fg="#888888", pady=5)
        coord_lbl.pack()
        
        return panel, coord_lbl

    def setup_ui(self):
        header = tk.Label(self.root, text="TargetTrackerVer4 Visualizer", font=("Outfit", 15, "bold"), bg="#121212", fg="#ffffff")
        header.pack(pady=15)
        
        self.frames_frame = tk.Frame(self.root, bg="#121212")
        self.frames_frame.pack(pady=10)
        
        self.ref_panel, self.ref_lbl = self.create_frame_slot(self.frames_frame, "Reference Target")
        self.curr_panel, self.curr_lbl = self.create_frame_slot(self.frames_frame, "Search Frame (GT)")
        self.pred_panel, self.pred_lbl = self.create_frame_slot(self.frames_frame, "Prediction & Heatmap")
        
        self.status_bar = tk.Label(self.root, text="Press [Space] for Next Sample", font=("Inter", 10), bg="#1c1c1c", fg="#aaaaaa", anchor="w", padx=15, pady=8)
        self.status_bar.pack(side="bottom", fill="x")
        
    def load_next_sample(self):
        try:
            import tensorflow as tf
            
            if self.current_batch_data is None or self.current_sample_idx >= self.current_batch_size - 1:
                if self.current_batch_data is not None:
                    self.current_pickle_idx = (self.current_pickle_idx + 1) % len(self.pickle_files)
                
                pickle_name = self.pickle_files[self.current_pickle_idx]
                pickle_path = os.path.join(self.dataset_dir, pickle_name)
                
                with open(pickle_path, "rb") as f:
                    self.current_batch_data = pickle.load(f)
                
                self.current_sample_idx = 0
                self.current_batch_size = len(self.current_batch_data)
            else:
                self.current_sample_idx += 1
                
            sample = self.current_batch_data[self.current_sample_idx]
            
            ref_stack = sample["reference_stack"]  # (16, 16, 16, 3)
            search_raw = sample["search_frame"]    # (H, W, 3)
            gt_heatmap = sample["ground_truth_heatmap"]
            meta = sample["metadata"]
            target_2d = meta["target_2d"]
            
            # Prepare Target Label
            # Scale target_2d to normalized coordinates
            h_raw, w_raw = search_raw.shape[:2]
            norm_x = target_2d[0] / w_raw
            norm_y = target_2d[1] / h_raw
            
            # Show the largest reference crop (layer 0) scaled up
            ref_layer_0 = ref_stack[0] # (16, 16, 3)
            ref_vis = cv2.resize(ref_layer_0, (256, 256), interpolation=cv2.INTER_NEAREST)
            self.tk_img_ref = ImageTk.PhotoImage(Image.fromarray(ref_vis))
            
            # Prepare inputs for model
            ref_tensor = tf.expand_dims(tf.cast(ref_stack, tf.float32) / 255.0, 0)
            
            search_256 = cv2.resize(search_raw, (256, 256), interpolation=cv2.INTER_LINEAR)
            search_tensor = tf.expand_dims(tf.cast(search_256, tf.float32) / 255.0, 0)
            
            # Predict
            pred = self.model([ref_tensor, search_tensor], training=False)
            pred_heatmap = pred.numpy()[0]  # (256, 256, 1)
            
            # Center of mass logic for sub-pixel prediction
            heatmap = pred_heatmap[:, :, 0]
            heatmap_thresh = np.where(heatmap > 0.1, heatmap, 0.0)
            total_mass = np.sum(heatmap_thresh)
            
            if total_mass > 1e-6:
                x_grid, y_grid = np.meshgrid(np.arange(256), np.arange(256))
                x_c = np.sum(x_grid * heatmap_thresh) / total_mass
                y_c = np.sum(y_grid * heatmap_thresh) / total_mass
                pred_norm = [x_c / 256.0, y_c / 256.0]
            else:
                pred_norm = [0.5, 0.5]
            
            self.tk_img_curr = self.process_and_draw(search_256, [norm_x, norm_y], "#00e6ff")
            
            # Overlay heatmap on search image
            heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(search_256, 0.6, heatmap_color, 0.4, 0)
            self.tk_img_pred = self.process_and_draw(overlay, pred_norm, "#33ff33")
            
            self.ref_panel.config(image=self.tk_img_ref)
            self.curr_panel.config(image=self.tk_img_curr)
            self.pred_panel.config(image=self.tk_img_pred)
            
            self.ref_lbl.config(text="Target Features")
            self.curr_lbl.config(text=f"GT: [{norm_x:.2f}, {norm_y:.2f}]", fg="#00e6ff")
            
            error = np.sqrt((pred_norm[0] - norm_x)**2 + (pred_norm[1] - norm_y)**2) * 256.0
            self.pred_lbl.config(text=f"Pred: [{pred_norm[0]:.2f}, {pred_norm[1]:.2f}]\nError: {error:.1f}px", fg="#33ff33")
            
            self.status_bar.config(text=f"Flight: {meta['flight_id']} | Frame: {meta['frame_idx']} | Dist: {meta['distance']:.1f}m | Press Space")
            
        except Exception as e:
            self.status_bar.config(text=f"Error: {e}")
            import traceback
            traceback.print_exc()

    def on_close(self):
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="dataset/")
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()
    
    root = tk.Tk()
    app = KerasFCNVisualizer(root, args.dataset_dir, args.model_path)
    root.mainloop()

if __name__ == "__main__":
    main()
