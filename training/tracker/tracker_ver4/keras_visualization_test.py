#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import argparse
import tkinter as tk
import pickle
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageTk

class KerasFCNVisualizer:
    def __init__(self, root, dataset_dir, model_path, threshold=0.5, min_blob_size=30):
        import tensorflow as tf
        
        self.threshold = threshold
        self.min_blob_size = min_blob_size
        
        print(f"Loading Keras TargetTrackerVer4 from {model_path}...")
        self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        
        self.dataset_dir = dataset_dir
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{self.dataset_dir}' does not exist.")
            
        self.pickle_files = [f for f in os.listdir(self.dataset_dir) if f.endswith('.pkl')]
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
        self.root.geometry("1150x500")
        self.root.configure(bg="#121212")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.setup_ui()
        
        self.root.bind("<space>", lambda e: self.load_next_sample())
        self.root.bind("<Escape>", lambda e: self.on_close())
        
        self.load_next_sample()

    def process_and_draw(self, img_256, coords_norm, circle_color):
        pil_img = Image.fromarray(img_256)
        if coords_norm is not None:
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
        
        coord_lbl = tk.Label(container, text="[N/A]", font=("Consolas", 10), bg="#121212", fg="#888888", pady=5, height=3)
        coord_lbl.pack()
        
        return panel, coord_lbl

    def setup_ui(self):
        header = tk.Label(self.root, text="TargetTrackerVer4 Visualizer", font=("Outfit", 15, "bold"), bg="#121212", fg="#ffffff")
        header.pack(pady=15)
        
        self.frames_frame = tk.Frame(self.root, bg="#121212")
        self.frames_frame.pack(pady=10)
        
        self.ref_panel, self.ref_lbl = self.create_frame_slot(self.frames_frame, "Reference Features")
        self.search_panel, self.search_lbl = self.create_frame_slot(self.frames_frame, "Search Frame")
        self.expected_heatmap_panel, self.expected_heatmap_lbl = self.create_frame_slot(self.frames_frame, "Expected Heatmap")
        self.predicted_heatmap_panel, self.predicted_heatmap_lbl = self.create_frame_slot(self.frames_frame, "Predicted Heatmap")
        
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
            
            ref_stack = sample["reference_stack"]  # (16, 16, 16, 1)
            search_raw = sample["search_frame"]    # (H, W, 1)
            gt_heatmap = sample["ground_truth_heatmap"]
            meta = sample["metadata"]
            target_2d = meta.get("target_2d")
            
            # Prepare Target Label
            # Scale target_2d to normalized coordinates
            h_raw, w_raw = search_raw.shape[:2]
            if target_2d is not None:
                norm_x = target_2d[0] / w_raw
                norm_y = target_2d[1] / h_raw
                norm_coords = [norm_x, norm_y]
                curr_lbl_text = f"GT: [{norm_x:.2f}, {norm_y:.2f}]"
                curr_lbl_fg = "#00e6ff"
            else:
                norm_coords = None
                curr_lbl_text = "GT: None (Negative)"
                curr_lbl_fg = "#ff3366"
            
            # Show the largest reference crop (layer 0) scaled up
            ref_layer_0 = ref_stack[0, :, :, 0] # (16, 16)
            ref_vis = cv2.resize(ref_layer_0, (256, 256), interpolation=cv2.INTER_NEAREST)
            self.tk_img_ref = ImageTk.PhotoImage(Image.fromarray(ref_vis))
            
            # Prepare inputs for model
            ref_tensor = tf.expand_dims(tf.cast(ref_stack, tf.float32) / 255.0, 0)
            
            search_256 = cv2.resize(search_raw[:, :, 0], (256, 256), interpolation=cv2.INTER_LINEAR)
            search_tensor = tf.expand_dims(tf.expand_dims(tf.cast(search_256, tf.float32) / 255.0, -1), 0)
            
            # Predict
            pred = self.model([ref_tensor, search_tensor], training=False)
            pred_heatmap = pred[0].numpy()[0]  # (256, 256, 1)
            pred_quality = pred[1].numpy()[0][0]  # scalar float
            
            # Local Refined Argmax Centroid Method for sub-pixel prediction
            heatmap = pred_heatmap[:, :, 0]
            
            # Apply threshold filter (noise gate)
            heatmap = np.where(heatmap >= self.threshold, heatmap, 0.0)
            
            # Apply connected component (blob size) filter
            if self.min_blob_size > 0:
                h, w = heatmap.shape
                visited = np.zeros((h, w), dtype=bool)
                for y in range(h):
                    for x in range(w):
                        if heatmap[y, x] > 0.0 and not visited[y, x]:
                            blob_pixels = []
                            queue = [(y, x)]
                            visited[y, x] = True
                            head = 0
                            while head < len(queue):
                                cy, cx = queue[head]
                                head += 1
                                blob_pixels.append((cy, cx))
                                
                                for dy in [-1, 0, 1]:
                                    for dx in [-1, 0, 1]:
                                        if dy == 0 and dx == 0:
                                            continue
                                        ny, nx = cy + dy, cx + dx
                                        if 0 <= ny < h and 0 <= nx < w:
                                            if heatmap[ny, nx] > 0.0 and not visited[ny, nx]:
                                                visited[ny, nx] = True
                                                queue.append((ny, nx))
                            
                            if len(blob_pixels) < self.min_blob_size:
                                for cy, cx in blob_pixels:
                                    heatmap[cy, cx] = 0.0
            
            flat_idx = np.argmax(heatmap)
            y_max, x_max = np.unravel_index(flat_idx, heatmap.shape)
            
            # Extract center of mass strictly in a local 5x5 neighborhood around the peak
            half_w = 2
            y_start = max(0, y_max - half_w)
            y_end = min(heatmap.shape[0], y_max + half_w + 1)
            x_start = max(0, x_max - half_w)
            x_end = min(heatmap.shape[1], x_max + half_w + 1)
            
            local_patch = heatmap[y_start:y_end, x_start:x_end]
            total_mass = np.sum(local_patch)
            
            if total_mass > 1e-6:
                y_grid, x_grid = np.mgrid[y_start:y_end, x_start:x_end]
                x_c = np.sum(x_grid * local_patch) / total_mass
                y_c = np.sum(y_grid * local_patch) / total_mass
                pred_norm = [x_c / 256.0, y_c / 256.0]
            else:
                pred_norm = [x_max / 256.0, y_max / 256.0]
            
            search_rgb = cv2.cvtColor(search_256, cv2.COLOR_GRAY2RGB)
            search_vis = search_rgb.copy()
            
            # Draw Expected target (cyan circle)
            if norm_coords is not None:
                cx = int(norm_coords[0] * 256.0)
                cy = int(norm_coords[1] * 256.0)
                cv2.circle(search_vis, (cx, cy), 6, (0, 230, 255), 2)
                cv2.circle(search_vis, (cx, cy), 2, (0, 230, 255), -1)
                
            # Draw Predicted target (green circle)
            pcx = int(pred_norm[0] * 256.0)
            pcy = int(pred_norm[1] * 256.0)
            cv2.circle(search_vis, (pcx, pcy), 6, (51, 255, 51), 2)
            cv2.circle(search_vis, (pcx, pcy), 2, (51, 255, 51), -1)
            
            self.tk_img_search = ImageTk.PhotoImage(Image.fromarray(search_vis))
            
            # Prepare clean raw expected heatmap
            gt_hm_256 = cv2.resize(gt_heatmap[:, :, 0].astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
            gt_heatmap_color = cv2.applyColorMap((gt_hm_256 * 255).astype(np.uint8), cv2.COLORMAP_JET)
            gt_heatmap_color_rgb = cv2.cvtColor(gt_heatmap_color, cv2.COLOR_BGR2RGB)
            self.tk_img_expected = ImageTk.PhotoImage(Image.fromarray(gt_heatmap_color_rgb))
            
            # Prepare clean raw predicted heatmap
            heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            self.tk_img_predicted = ImageTk.PhotoImage(Image.fromarray(heatmap_color_rgb))
            
            self.ref_panel.config(image=self.tk_img_ref)
            self.search_panel.config(image=self.tk_img_search)
            self.expected_heatmap_panel.config(image=self.tk_img_expected)
            self.predicted_heatmap_panel.config(image=self.tk_img_predicted)
            
            self.ref_lbl.config(text="Target Features")
            self.search_lbl.config(text=curr_lbl_text, fg=curr_lbl_fg)
            
            if target_2d is not None:
                error = np.sqrt((pred_norm[0] - norm_x)**2 + (pred_norm[1] - norm_y)**2) * 256.0
                error_str = f"Error: {error:.1f}px"
            else:
                error_str = "Error: N/A"
            
            self.expected_heatmap_lbl.config(text="GT Heatmap")
            self.predicted_heatmap_lbl.config(text=f"Pred: [{pred_norm[0]:.2f}, {pred_norm[1]:.2f}]\n{error_str}\nQuality: {pred_quality:.2f}", fg="#33ff33")
            
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
    parser.add_argument("--threshold", type=float, default=0.5, help="Heatmap noise threshold")
    parser.add_argument("--min_blob_size", type=int, default=30, help="Minimum connected component size to keep")
    args = parser.parse_args()
    
    root = tk.Tk()
    app = KerasFCNVisualizer(
        root, 
        args.dataset_dir, 
        args.model_path, 
        threshold=args.threshold, 
        min_blob_size=args.min_blob_size
    )
    root.mainloop()

if __name__ == "__main__":
    main()
