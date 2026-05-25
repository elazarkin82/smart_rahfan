#!/usr/bin/env python3
import os
import sys
import shutil
import pickle
import argparse
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageTk

# Add parent directory to path to allow importing tracker module if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracker.tracker_model import TargetTracker

class DatasetVisualizer:
    def __init__(self, root, images_path):
        self.root = root
        self.images_path = images_path
        self.temp_dir = "temp_visual_test"
        
        # Configure window
        self.root.title("Recursive Target Tracker - Visual Dataset Inspector")
        self.root.geometry("860x420")
        self.root.configure(bg="#121212")  # Sleek dark mode background
        self.root.resizable(False, False)
        
        # Setup temporary directories
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Window closing handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Build UI layout
        self.setup_ui()
        
        # Bind keyboard events
        self.root.bind("<space>", lambda e: self.load_next_sample())
        self.root.bind("<Escape>", lambda e: self.on_close())
        
        # Load first sample
        self.load_next_sample()
        
    def setup_ui(self):
        # 1. Main Header
        header = tk.Label(
            self.root, 
            text="Recursive Target Tracker - Dataset Visual Inspector", 
            font=("Outfit", 16, "bold"), 
            bg="#121212", 
            fg="#ffffff"
        )
        header.pack(pady=15)
        
        # 2. Frames Container (Horizontal Layout)
        self.frames_frame = tk.Frame(self.root, bg="#121212")
        self.frames_frame.pack(pady=10)
        
        # 3. Individual Frame Slots
        self.hist_panel = self.create_frame_slot(self.frames_frame, "Historical Frame (Warped)")
        self.prev_panel = self.create_frame_slot(self.frames_frame, "Previous Frame (Warped)")
        self.curr_panel = self.create_frame_slot(self.frames_frame, "Current Frame (Label Target)")
        
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
        
    def create_frame_slot(self, parent, label_text):
        container = tk.Frame(parent, bg="#121212", padx=10)
        container.pack(side="left")
        
        # Title above image
        lbl = tk.Label(
            container, 
            text=label_text, 
            font=("Inter", 11, "bold"), 
            bg="#121212", 
            fg="#00e6ff" if "Current" in label_text else "#e0e0e0"
        )
        lbl.pack(pady=5)
        
        # Canvas/Panel for image
        panel = tk.Label(container, bg="#1a1a1a", bd=1, relief="solid", width=256, height=256)
        panel.pack()
        
        return panel
        
    def process_and_draw(self, frame_np, coords_np, circle_color):
        # frame_np is shape (256, 256, 1), scaled in [0, 1]
        img_8u = (frame_np[:, :, 0] * 255.0).astype(np.uint8)
        
        # Convert to RGB to allow drawing a colored circle
        img_rgb = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Draw red circle on the target point
        draw = ImageDraw.Draw(pil_img)
        r = 6  # Circle radius
        x_px = int(coords_np[0] * 256.0)
        y_px = int(coords_np[1] * 256.0)
        
        # Draw outer glowing circle and center dot
        draw.ellipse([x_px - r, y_px - r, x_px + r, y_px + r], outline=circle_color, width=2)
        draw.ellipse([x_px - 2, y_px - 2, x_px + 2, y_px + 2], fill=circle_color)
        
        return ImageTk.PhotoImage(pil_img)

    def load_next_sample(self):
        self.status_bar.config(text="Generating next sample via generate_dataset()...", fg="#ffcc00")
        self.root.update_idletasks()
        
        try:
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
            curr_coord = targets[0] # expected output
            
            # Process and draw circles
            self.tk_img_hist = self.process_and_draw(hist_frame, hist_coord, "#ff3333")  # Red for hist
            self.tk_img_prev = self.process_and_draw(prev_frame, prev_coord, "#ff9900")  # Orange for prev
            self.tk_img_curr = self.process_and_draw(curr_frame, curr_coord, "#00e6ff")  # Cyan for curr
            
            # Update GUI panels
            self.hist_panel.config(image=self.tk_img_hist)
            self.prev_panel.config(image=self.tk_img_prev)
            self.curr_panel.config(image=self.tk_img_curr)
            
            # Delete file to save space and prepare for next call
            if os.path.exists(pickle_path):
                os.remove(pickle_path)
                
            self.status_bar.config(
                text=f"Press [Space] for Next Sample  |  Press [Esc] to Exit  |  Source: {self.images_path}",
                fg="#aaaaaa"
            )
            
        except Exception as e:
            self.status_bar.config(text=f"Error: {e}", fg="#ff3333")
            messagebox.showerror("Dataset Error", f"Failed to generate sample:\n{e}")

    def on_close(self):
        # Cleanup temporary visual files
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser(description="TargetTracker Visual Dataset Inspector GUI")
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
    app = DatasetVisualizer(root, args.images_path)
    root.mainloop()

if __name__ == "__main__":
    main()
