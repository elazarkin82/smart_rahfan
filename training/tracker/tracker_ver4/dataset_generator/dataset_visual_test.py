#!/usr/bin/env python3
import os
import sys
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageTk

def load_config(path="pipeline_config.json"):
    """Loads JSON configuration safely."""
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to load config from {path}: {e}")
    return {}

def get_crop(image, center_x, center_y, size):
    """Extracts a square crop of 'size' around the center, padding with edge replication if out of bounds."""
    h, w = image.shape[:2]
    half = size / 2.0
    
    x1 = int(round(center_x - half))
    x2 = int(round(center_x + half))
    y1 = int(round(center_y - half))
    y2 = int(round(center_y + half))
    
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Extract
    crop = image[y1:y2, x1:x2].copy()
    
    # Pad if we hit the edge of the screen
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
        
    return crop

def build_reference_stack(image, center, num_layers, max_size, min_size, target_size):
    """Builds a spatial pyramid reference crop stack on the fly (for raw cache mode)."""
    sizes = np.linspace(max_size, min_size, num_layers)
    stack_layers = []
    
    for sz in sizes:
        crop = get_crop(image, center[0], center[1], sz)
        resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        stack_layers.append(resized)
        
    stack = np.stack(stack_layers, axis=0).astype(np.uint8)
    return np.expand_dims(stack, axis=-1)

class DatasetVisualizer:
    def __init__(self, root, cache_dir=None, dataset_dir=None):
        self.root = root
        self.root.title("TargetTrackerVer4 - Grayscale Dataset Visualizer")
        self.root.geometry("1120x720")
        self.root.configure(bg="#121212")
        self.root.resizable(True, True)
        
        # Load pipeline config relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "pipeline_config.json")
        self.config = load_config(config_path)
        
        # Setup paths (with fallback logic)
        self.cache_dir = cache_dir or self.config.get("generation", {}).get("cache_dir", "cache")
        self.dataset_dir = dataset_dir or self.config.get("compiler", {}).get("compiled_dir", "compiled")
        
        if not os.path.isabs(self.cache_dir):
            self.cache_dir = os.path.abspath(os.path.join(script_dir, self.cache_dir))
        if not os.path.isabs(self.dataset_dir):
            self.dataset_dir = os.path.abspath(os.path.join(script_dir, self.dataset_dir))
            
        print(f"[Visualizer] Cache directory: {self.cache_dir}")
        print(f"[Visualizer] Compiled Frames directory: {self.dataset_dir}")
        
        # Discover files
        self.raw_files = []
        if os.path.exists(self.cache_dir):
            self.raw_files = sorted([f for f in os.listdir(self.cache_dir) if f.startswith("flight_") and f.endswith(".pkl")])
            
        self.compiled_files = []
        if os.path.exists(self.dataset_dir):
            if os.path.exists(os.path.join(self.dataset_dir, "dataset.h5")):
                self.compiled_files.append("dataset.h5")
            self.compiled_files.extend(sorted([
                f for f in os.listdir(self.dataset_dir) 
                if f.endswith(".pkl") and os.path.isfile(os.path.join(self.dataset_dir, f))
            ]))
            
        print(f"[Visualizer] Found {len(self.raw_files)} raw flights in cache.")
        print(f"[Visualizer] Found {len(self.compiled_files)} compiled files in dataset.")
        
        # State
        self.h5_file = None
        if self.raw_files:
            self.mode = "raw"
        elif self.compiled_files:
            self.mode = "compiled"
        else:
            self.mode = "raw"
            
        self.current_file_idx = 0
        self.current_frame_idx = 0
        self.playing = False
        self.play_delay_ms = 80  # Default playback speed (~12.5 FPS)
        self.loaded_data = None
        self.search_tk_image = None
        self.heatmap_tk_image = None
        self.ref_tk_images = []
        
        # UI Setup
        self.setup_ui()
        
        # Bind keys
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Up>", lambda e: self.next_file())
        self.root.bind("<Down>", lambda e: self.prev_file())
        self.root.bind("<m>", lambda e: self.toggle_mode_hotkey())
        self.root.bind("<M>", lambda e: self.toggle_mode_hotkey())
        self.root.bind("<Delete>", lambda e: self.delete_current_file())
        self.root.bind("<Escape>", lambda e: self.root.quit())
        
        # Initialize first load
        self.populate_file_list()
        self.load_file()

    def setup_ui(self):
        # Premium Dark styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background="#121212", foreground="#ffffff")
        style.configure("TLabel", background="#121212", foreground="#ffffff")
        style.configure("TFrame", background="#121212")
        style.configure("TButton", background="#2a2a2a", foreground="#ffffff", borderwidth=0, focuscolor="none")
        style.map("TButton", background=[("active", "#3a3a3a")])
        style.configure("Delete.TButton", background="#8b0000", foreground="#ffffff", borderwidth=0, focuscolor="none")
        style.map("Delete.TButton", background=[("active", "#b22222")])
        style.configure("TCombobox", fieldbackground="#1e1e1e", background="#2d2d2d", foreground="#ffffff")
        
        # 1. Header Frame
        header_frame = tk.Frame(self.root, bg="#1e1e1e", height=60)
        header_frame.pack(side="top", fill="x")
        header_frame.pack_propagate(False)
        
        title_lbl = tk.Label(header_frame, text="TRACKER V4 DATASET INSPECTOR", font=("Outfit", 12, "bold"), bg="#1e1e1e", fg="#00e6ff")
        title_lbl.pack(side="left", padx=15)
        
        # Mode selector
        mode_lbl = tk.Label(header_frame, text="Mode:", bg="#1e1e1e", fg="#aaaaaa")
        mode_lbl.pack(side="left", padx=(15, 5))
        
        self.mode_combo = ttk.Combobox(header_frame, values=["Raw Flight Cache (flight_*.pkl)", "Compiled Frames (compiled/)"], state="readonly", width=30)
        self.mode_combo.current(0 if self.mode == "raw" else 1)
        self.mode_combo.pack(side="left", padx=5)
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_mode_changed)
        
        # File selector
        file_lbl = tk.Label(header_frame, text="File:", bg="#1e1e1e", fg="#aaaaaa")
        file_lbl.pack(side="left", padx=(15, 5))
        
        self.file_combo = ttk.Combobox(header_frame, state="readonly", width=25)
        self.file_combo.pack(side="left", padx=5)
        self.file_combo.bind("<<ComboboxSelected>>", self.on_file_changed)
        
        # 2. Main split view
        main_view = tk.Frame(self.root, bg="#121212")
        main_view.pack(side="top", fill="both", expand=True, padx=15, pady=15)
        
        # Left main display area
        self.left_panel = tk.Frame(main_view, bg="#121212", width=730, height=520)
        self.left_panel.pack(side="left", fill="both", expand=True)
        self.left_panel.pack_propagate(False)
        
        # Two side-by-side panels for separate raw displays
        self.search_image_label = tk.Label(self.left_panel, bg="#1a1a1a", bd=1, relief="solid")
        self.search_image_label.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        self.heatmap_image_label = tk.Label(self.left_panel, bg="#1a1a1a", bd=1, relief="solid")
        self.heatmap_image_label.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Right reference crop area
        right_panel = tk.Frame(main_view, bg="#1a1a1a", width=350, bd=1, relief="solid")
        right_panel.pack(side="right", fill="y", padx=5, pady=5)
        right_panel.pack_propagate(False)
        
        ref_title = tk.Label(right_panel, text="Multi-Scale Reference Stack (16 Layers)", font=("Inter", 11, "bold"), bg="#1a1a1a", fg="#ffffff")
        ref_title.pack(pady=12)
        
        # 4x4 Grid for reference crops
        self.grid_frame = tk.Frame(right_panel, bg="#1a1a1a")
        self.grid_frame.pack(pady=10)
        
        self.ref_stack_labels = []
        for r in range(4):
            for c in range(4):
                idx = r * 4 + c
                cell_frame = tk.Frame(self.grid_frame, bg="#262626", width=68, height=68, bd=1, relief="solid")
                cell_frame.grid(row=r, column=c, padx=4, pady=4)
                cell_frame.grid_propagate(False)
                
                lbl = tk.Label(cell_frame, text=f"L{idx}", font=("Consolas", 9), bg="#262626", fg="#777777")
                lbl.pack(fill="both", expand=True)
                self.ref_stack_labels.append(lbl)
                
        # Status details inside right panel
        details_frame = tk.LabelFrame(right_panel, text=" Metadata Details ", bg="#1a1a1a", fg="#00e6ff", font=("Inter", 9, "bold"), padx=10, pady=10)
        details_frame.pack(fill="x", side="bottom", padx=15, pady=15)
        
        self.detail_flight_lbl = tk.Label(details_frame, text="Flight: N/A", anchor="w", bg="#1a1a1a", fg="#cccccc", font=("Consolas", 9))
        self.detail_flight_lbl.pack(fill="x", pady=2)
        
        self.detail_frame_lbl = tk.Label(details_frame, text="Frame: N/A", anchor="w", bg="#1a1a1a", fg="#cccccc", font=("Consolas", 9))
        self.detail_frame_lbl.pack(fill="x", pady=2)
        
        self.detail_pos2d_lbl = tk.Label(details_frame, text="Target 2D: N/A", anchor="w", bg="#1a1a1a", fg="#cccccc", font=("Consolas", 9))
        self.detail_pos2d_lbl.pack(fill="x", pady=2)
        
        self.detail_dist_lbl = tk.Label(details_frame, text="Distance: N/A", anchor="w", bg="#1a1a1a", fg="#cccccc", font=("Consolas", 9))
        self.detail_dist_lbl.pack(fill="x", pady=2)
        
        self.detail_pos3d_lbl = tk.Label(details_frame, text="Target 3D: N/A", anchor="w", bg="#1a1a1a", fg="#cccccc", font=("Consolas", 9))
        self.detail_pos3d_lbl.pack(fill="x", pady=2)
        
        # 3. HUD status strip
        self.info_label = tk.Label(self.root, text="Loading dataset files...", font=("Consolas", 10), bg="#1c1c1c", fg="#00ff66", height=2, anchor="w", padx=15)
        self.info_label.pack(side="top", fill="x")
        
        # 4. Controls Frame
        controls_bar = tk.Frame(self.root, bg="#1e1e1e", height=70)
        controls_bar.pack(side="bottom", fill="x")
        controls_bar.pack_propagate(False)
        
        # Navigation buttons
        btn_font = ("Inter", 9, "bold")
        
        self.btn_prev_file = ttk.Button(controls_bar, text="⏮ Prev Flight [↓]", width=15, command=self.prev_file)
        self.btn_prev_file.pack(side="left", padx=15, pady=20)
        
        self.btn_prev_frame = ttk.Button(controls_bar, text="◀ Prev Frame [←]", width=15, command=self.prev_frame)
        self.btn_prev_frame.pack(side="left", padx=5)
        
        self.play_button = ttk.Button(controls_bar, text=" ▶ Play [Space] ", width=15, command=self.toggle_play)
        self.play_button.pack(side="left", padx=5)
        
        self.btn_next_frame = ttk.Button(controls_bar, text="Next Frame [→] ▶", width=15, command=self.next_frame)
        self.btn_next_frame.pack(side="left", padx=5)
        
        self.btn_next_file = ttk.Button(controls_bar, text="Next Flight [↑] ⏭", width=15, command=self.next_file)
        self.btn_next_file.pack(side="left", padx=5)
        
        self.btn_delete = ttk.Button(controls_bar, text="🗑 Delete [Del]", width=15, style="Delete.TButton", command=self.delete_current_file)
        self.btn_delete.pack(side="left", padx=5)
        
        # Playback speed slider
        speed_frame = tk.Frame(controls_bar, bg="#1e1e1e")
        speed_frame.pack(side="right", padx=20, pady=10)
        
        self.speed_label = tk.Label(speed_frame, text="Frame Delay: 80 ms", font=("Inter", 9), bg="#1e1e1e", fg="#cccccc")
        self.speed_label.pack(side="top")
        
        self.speed_slider = tk.Scale(speed_frame, from_=20, to=400, orient="horizontal", length=180, bg="#1e1e1e", fg="#ffffff", highlightthickness=0, resolution=10, command=self.on_speed_changed)
        self.speed_slider.set(80)
        self.speed_slider.pack(side="bottom")

    def populate_file_list(self):
        files = self.raw_files if self.mode == "raw" else self.compiled_files
        self.file_combo['values'] = files
        if files:
            if self.current_file_idx >= len(files):
                self.current_file_idx = 0
            self.file_combo.current(self.current_file_idx)
        else:
            self.file_combo.set("No files found")

    def on_mode_changed(self, event=None):
        selection = self.mode_combo.get()
        new_mode = "raw" if "Raw" in selection else "compiled"
        if new_mode != self.mode:
            self.mode = new_mode
            self.current_file_idx = 0
            self.populate_file_list()
            self.load_file()

    def on_file_changed(self, event=None):
        files = self.raw_files if self.mode == "raw" else self.compiled_files
        if not files:
            return
        selected_file = self.file_combo.get()
        if selected_file in files:
            self.current_file_idx = files.index(selected_file)
            self.load_file()

    def load_file(self):
        self.playing = False
        self.play_button.config(text=" ▶ Play [Space] ")
        
        # Close previous H5 file if open
        if hasattr(self, 'h5_file') and self.h5_file:
            try:
                self.h5_file.close()
            except Exception:
                pass
            self.h5_file = None
            
        files = self.raw_files if self.mode == "raw" else self.compiled_files
        if not files:
            self.loaded_data = None
            self.current_frame_idx = 0
            self.update_ui()
            return
            
        if self.current_file_idx >= len(files):
            self.current_file_idx = 0
        elif self.current_file_idx < 0:
            self.current_file_idx = len(files) - 1
            
        filename = files[self.current_file_idx]
        dir_path = self.cache_dir if self.mode == "raw" else self.dataset_dir
        filepath = os.path.join(dir_path, filename)
        
        try:
            if filepath.endswith(".h5"):
                import h5py
                self.h5_file = h5py.File(filepath, 'r')
                self.loaded_data = range(self.h5_file["reference_stack"].shape[0])
            else:
                with open(filepath, 'rb') as f:
                    self.loaded_data = pickle.load(f)
            self.current_frame_idx = 0
            self.file_combo.current(self.current_file_idx)
            print(f"[Visualizer] Loaded {self.mode} file: {filename} ({len(self.loaded_data)} frames/samples)")
        except Exception as e:
            print(f"[Error] Failed to load {filepath}: {e}")
            messagebox.showerror("Error Loading File", f"Could not open/parse file:\n{filepath}\n\nError: {e}")
            self.loaded_data = None
            
        self.update_ui()

    def update_ui(self):
        if not self.loaded_data:
            self.search_image_label.config(image='', text="No Data Loaded. Please check paths.")
            self.heatmap_image_label.config(image='', text="No heatmap loaded.")
            self.info_label.config(text="No files found to inspect.", fg="#ff3366")
            self.detail_flight_lbl.config(text="Flight: N/A")
            self.detail_frame_lbl.config(text="Frame: N/A")
            self.detail_pos2d_lbl.config(text="Target 2D: N/A")
            self.detail_dist_lbl.config(text="Distance: N/A")
            self.detail_pos3d_lbl.config(text="Target 3D: N/A")
            for i in range(16):
                self.ref_stack_labels[i].config(image='', text=f"L{i}")
            return
            
        num_frames = len(self.loaded_data)
        if self.current_frame_idx >= num_frames:
            self.current_frame_idx = 0
        elif self.current_frame_idx < 0:
            self.current_frame_idx = num_frames - 1
            
        if self.mode == "compiled" and hasattr(self, 'h5_file') and self.h5_file:
            idx = self.current_frame_idx
            ref_stack = self.h5_file["reference_stack"][idx]
            search_frame = self.h5_file["search_frame"][idx]
            gt_heatmap = self.h5_file["ground_truth_heatmap"][idx]
            gt_quality = self.h5_file["ground_truth_quality"][idx][0]
            
            # Find peak of heatmap
            hm_np = gt_heatmap[:, :, 0]
            if np.max(hm_np) > 0.1:
                py, px = np.unravel_index(np.argmax(hm_np), hm_np.shape)
                target_2d = (px, py)
            else:
                target_2d = None
                
            sample = {
                "reference_stack": ref_stack,
                "search_frame": search_frame,
                "ground_truth_heatmap": gt_heatmap,
                "metadata": {
                    "target_2d": target_2d,
                    "distance": 0.0,
                    "quality": gt_quality
                }
            }
            self.render_compiled(sample)
        else:
            sample = self.loaded_data[self.current_frame_idx]
            if self.mode == "raw":
                self.render_raw(sample)
            else:
                self.render_compiled(sample)

    def render_raw(self, sample):
        img_gray = sample["image_gray"]
        h_raw, w_raw = img_gray.shape[:2]
        
        # Convert Grayscale to RGB for color decorations
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        # Draw neon-pink crosshairs
        target_2d = sample["target_2d"]
        px, py = int(target_2d[0]), int(target_2d[1])
        
        color = (255, 51, 102) # #ff3366 Neon Pink
        cv2.circle(img_rgb, (px, py), 15, color, 2)
        cv2.circle(img_rgb, (px, py), 2, color, -1)
        # Reticle crosshair ticks
        cv2.line(img_rgb, (px - 22, py), (px - 8, py), color, 1)
        cv2.line(img_rgb, (px + 8, py), (px + 22, py), color, 1)
        cv2.line(img_rgb, (px, py - 22), (px, py - 8), color, 1)
        cv2.line(img_rgb, (px, py + 8), (px, py + 22), color, 1)
        
        # Fit image dynamically into left panel, retaining aspect ratio if possible
        panel_w = self.search_image_label.winfo_width()
        panel_h = self.search_image_label.winfo_height()
        if panel_w < 100 or panel_h < 100: # not mapped yet
            panel_w, panel_h = 480, 480
            
        # Scale to match display nicely
        scale = min(panel_w / w_raw, panel_h / h_raw)
        disp_w = int(w_raw * scale)
        disp_h = int(h_raw * scale)
        
        if disp_w > 0 and disp_h > 0:
            img_resized = cv2.resize(img_rgb, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img_rgb
            
        pil_img = Image.fromarray(img_resized)
        self.search_tk_image = ImageTk.PhotoImage(image=pil_img)
        self.search_image_label.config(image=self.search_tk_image, text='')
        
        # In raw mode, expected heatmap is not compiled yet
        self.heatmap_image_label.config(image='', text="[Raw Mode - No Compiled Heatmap]")
        
        # Metadata
        dist = sample["distance_to_target"]
        t3d = sample.get("target_3d", [0.0, 0.0, 0.0])
        filename = self.raw_files[self.current_file_idx]
        
        hud_text = f"Flight Cache  |  File: {filename}  |  Frame: {self.current_frame_idx} / {len(self.loaded_data)-1}  |  " \
                   f"Target 2D: [{px}, {py}]  |  Dist: {dist:.1f}m"
        self.info_label.config(text=hud_text, fg="#00ff66")
        
        self.detail_flight_lbl.config(text=f"Flight: {filename}")
        self.detail_frame_lbl.config(text=f"Frame: {self.current_frame_idx} / {len(self.loaded_data)-1}")
        self.detail_pos2d_lbl.config(text=f"Target 2D: [{px}, {py}]")
        self.detail_dist_lbl.config(text=f"Distance: {dist:.2f} m")
        self.detail_pos3d_lbl.config(text=f"Target 3D: [{t3d[0]:.2f}, {t3d[1]:.2f}, {t3d[2]:.2f}]")
        
        # Build Reference Stack on the fly for raw cache visualizer (using Frame 0 as anchor)
        ref_frame = self.loaded_data[0]
        ref_gray = ref_frame["image_gray"]
        ref_target = ref_frame["target_2d"]
        
        layers = self.config.get("compiler", {}).get("stack_layers", 16)
        max_sz = ref_frame.get("crop_max_size", self.config.get("compiler", {}).get("crop_max_size", 128))
        min_sz = ref_frame.get("crop_min_size", self.config.get("compiler", {}).get("crop_min_size", 4))
        tgt_sz = self.config.get("compiler", {}).get("stack_target_size", 16)
        
        ref_stack = build_reference_stack(ref_gray, ref_target, layers, max_sz, min_sz, tgt_sz)
        self.display_ref_stack(ref_stack)

    def render_compiled(self, sample):
        # 1. Base search image
        search_gray = sample["search_frame"][:, :, 0]
        if search_gray.dtype != np.uint8:
            search_gray = (search_gray * 255.0).astype(np.uint8)
        h_s, w_s = search_gray.shape[:2]
        search_rgb = cv2.cvtColor(search_gray, cv2.COLOR_GRAY2RGB)
        
        # 2. Extract ground truth heatmap
        gt_heatmap = sample["ground_truth_heatmap"][:, :, 0].astype(np.float32)
        if np.max(gt_heatmap) <= 1.001:
            heatmap_scaled = (gt_heatmap * 255.0).astype(np.uint8)
        else:
            heatmap_scaled = gt_heatmap.astype(np.uint8)
            
        # Colorize Heatmap to Jet
        heatmap_color = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Target from metadata
        meta = sample.get("metadata", {})
        target_2d = meta.get("target_2d")
        
        # Prepare clean Search frame with target ring
        search_vis = search_rgb.copy()
        if target_2d is not None:
            px, py = int(target_2d[0]), int(target_2d[1])
            # Draw cyan target ring on search crop
            cv2.circle(search_vis, (px, py), 8, (0, 230, 255), 1)
            cv2.circle(search_vis, (px, py), 1, (0, 230, 255), -1)
            target_str = f"[{px}, {py}]"
        else:
            target_str = "None (Negative Sample)"
            
        # Scale and render clean search frame
        panel_w = self.search_image_label.winfo_width()
        panel_h = self.search_image_label.winfo_height()
        if panel_w < 100 or panel_h < 100:
            panel_w, panel_h = 256, 256
            
        scale_s = min(panel_w / w_s, panel_h / h_s)
        disp_w_s = int(w_s * scale_s)
        disp_h_s = int(h_s * scale_s)
        
        if disp_w_s > 0 and disp_h_s > 0:
            search_resized = cv2.resize(search_vis, (disp_w_s, disp_h_s), interpolation=cv2.INTER_LINEAR)
            heatmap_resized = cv2.resize(heatmap_color, (disp_w_s, disp_h_s), interpolation=cv2.INTER_LINEAR)
        else:
            search_resized = search_vis
            heatmap_resized = heatmap_color
            
        pil_search = Image.fromarray(search_resized)
        self.search_tk_image = ImageTk.PhotoImage(image=pil_search)
        self.search_image_label.config(image=self.search_tk_image, text='')
        
        # Draw quality score value on heatmap image for validation inspection
        cv2.putText(heatmap_resized, f"Quality: {meta.get('quality', 1.0):.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        pil_heatmap = Image.fromarray(heatmap_resized)
        self.heatmap_tk_image = ImageTk.PhotoImage(image=pil_heatmap)
        self.heatmap_image_label.config(image=self.heatmap_tk_image, text='')
        
        # Metadata update
        filename = self.compiled_files[self.current_file_idx]
        dist = meta.get("distance", 0.0)
        
        hud_text = f"Compiled Frames  |  File: {filename}  |  Sample: {self.current_frame_idx+1} / {len(self.loaded_data)}  |  " \
                   f"Target 2d: {target_str}  |  Dist: {dist:.1f}m"
        self.info_label.config(text=hud_text, fg="#00e6ff")
        
        self.detail_flight_lbl.config(text=f"File: {filename}")
        self.detail_frame_lbl.config(text=f"Sample: {self.current_frame_idx+1} / {len(self.loaded_data)}")
        self.detail_pos2d_lbl.config(text=f"Target 2D: {target_str}")
        self.detail_dist_lbl.config(text=f"Distance: {dist:.2f} m")
        self.detail_pos3d_lbl.config(text="Target 3D: [Compiled]")
        
        # Display the pre-built reference stack in sample
        ref_stack = sample["reference_stack"]
        self.display_ref_stack(ref_stack)

    def display_ref_stack(self, ref_stack):
        # ref_stack is shape: (num_layers, size, size, 1) in legacy, or (1, size, size, num_layers) in new
        if ref_stack.ndim == 4 and ref_stack.shape[0] == 1:
            num_layers = ref_stack.shape[3]
        else:
            num_layers = ref_stack.shape[0]
            
        self.ref_tk_images = [] # Prevent garbage collection
        
        for i in range(16):
            if i < num_layers:
                if ref_stack.ndim == 4 and ref_stack.shape[0] == 1:
                    layer_gray = ref_stack[0, :, :, i]
                else:
                    layer_gray = ref_stack[i, :, :, 0]
                    
                # Scale up to 64x64 using Nearest Neighbor to see individual pixels clearly
                layer_resized = cv2.resize(layer_gray, (64, 64), interpolation=cv2.INTER_NEAREST)
                
                # Check if it is float or uint8
                if layer_resized.dtype != np.uint8:
                    layer_resized = (layer_resized * 255.0).astype(np.uint8)
                    
                # Convert to RGB PIL Image
                pil_img = Image.fromarray(cv2.cvtColor(layer_resized, cv2.COLOR_GRAY2RGB))
                tk_img = ImageTk.PhotoImage(image=pil_img)
                self.ref_tk_images.append(tk_img)
                
                self.ref_stack_labels[i].config(image=tk_img, text='')
            else:
                self.ref_stack_labels[i].config(image='', text=f"L{i}")

    def toggle_play(self):
        if self.playing:
            self.playing = False
            self.play_button.config(text=" ▶ Play [Space] ")
        else:
            if not self.loaded_data:
                return
            self.playing = True
            self.play_button.config(text=" ❚❚ Pause [Space] ")
            self.playback_loop()

    def playback_loop(self):
        if not self.playing or not self.loaded_data:
            return
            
        self.current_frame_idx = (self.current_frame_idx + 1) % len(self.loaded_data)
        self.update_ui()
        self.root.after(self.play_delay_ms, self.playback_loop)

    def next_frame(self):
        if not self.loaded_data: return
        self.current_frame_idx = (self.current_frame_idx + 1) % len(self.loaded_data)
        self.update_ui()

    def prev_frame(self):
        if not self.loaded_data: return
        self.current_frame_idx = (self.current_frame_idx - 1) % len(self.loaded_data)
        self.update_ui()

    def next_file(self):
        files = self.raw_files if self.mode == "raw" else self.compiled_files
        if not files: return
        self.current_file_idx = (self.current_file_idx + 1) % len(files)
        self.load_file()

    def prev_file(self):
        files = self.raw_files if self.mode == "raw" else self.compiled_files
        if not files: return
        self.current_file_idx = (self.current_file_idx - 1) % len(files)
        self.load_file()

    def toggle_mode_hotkey(self):
        new_mode = "compiled" if self.mode == "raw" else "raw"
        new_selection = "Compiled Frames (compiled/)" if new_mode == "compiled" else "Raw Flight Cache (flight_*.pkl)"
        self.mode_combo.set(new_selection)
        self.on_mode_changed()

    def on_speed_changed(self, val):
        self.play_delay_ms = int(float(val))
        self.speed_label.config(text=f"Frame Delay: {self.play_delay_ms} ms")

    def delete_current_file(self):
        files = self.raw_files if self.mode == "raw" else self.compiled_files
        if not files or not self.loaded_data:
            return
            
        filename = files[self.current_file_idx]
        
        # Safety Check: Do not allow deletion of dataset.h5 from this visualizer.
        if filename == "dataset.h5":
            messagebox.showwarning(
                "Cannot Delete Dataset",
                "You cannot delete the compiled dataset file 'dataset.h5' directly from here.\n\n"
                "To remove bad flights:\n"
                "1. Switch Mode to 'Raw Flight Cache'.\n"
                "2. Delete the bad raw flight(s).\n"
                "3. Re-run the compiler to generate a clean 'dataset.h5'."
            )
            return
            
        dir_path = self.cache_dir if self.mode == "raw" else self.dataset_dir
        filepath = os.path.join(dir_path, filename)
        
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete this flight file?\n\n{filename}\n\nThis action cannot be undone."
        )
        if not confirm:
            return
            
        # Close H5 file if it is currently open
        if hasattr(self, 'h5_file') and self.h5_file:
            try:
                self.h5_file.close()
            except Exception:
                pass
            self.h5_file = None
            
        try:
            os.remove(filepath)
            print(f"[Visualizer] Deleted file: {filepath}")
            
            # Remove from local list
            files.pop(self.current_file_idx)
            self.file_combo['values'] = files
            
            if not files:
                self.loaded_data = None
                self.current_file_idx = 0
            else:
                if self.current_file_idx >= len(files):
                    self.current_file_idx = len(files) - 1
                self.file_combo.current(self.current_file_idx)
                
            self.load_file()
            messagebox.showinfo("Deleted", f"Successfully deleted {filename}")
        except Exception as e:
            print(f"[Error] Failed to delete {filepath}: {e}")
            messagebox.showerror("Error Deleting File", f"Could not delete file:\n{filepath}\n\nError: {e}")

def main():
    parser = argparse.ArgumentParser(description="TargetTrackerVer4 Dataset & Cache Visualizer")
    parser.add_argument("--cache_dir", default=None, help="Directory containing raw flight_.pkl files")
    parser.add_argument("--dataset_dir", default=None, help="Directory containing compiled PKL files (train_*.pkl, batch_*.pkl, etc.)")
    args = parser.parse_args()
    
    root = tk.Tk()
    app = DatasetVisualizer(root, cache_dir=args.cache_dir, dataset_dir=args.dataset_dir)
    root.mainloop()

if __name__ == "__main__":
    main()
