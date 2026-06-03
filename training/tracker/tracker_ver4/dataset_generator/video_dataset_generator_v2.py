#!/usr/bin/env python3
import os
import sys
import argparse
import pickle
import glob
import math
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

class VideoDatasetGeneratorV2:
    def __init__(self, root, video_dir, cache_dir):
        self.root = root
        self.video_dir = video_dir
        self.cache_dir = cache_dir
        
        # Ensure directories exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Window properties
        self.root.title("TargetTrackerVer4 - Video Dataset Generator V2 (Step-by-Step)")
        self.root.configure(bg="#121212")
        self.root.state('normal')
        
        # Maximize screen
        try:
            self.root.attributes('-zoomed', True)
        except Exception:
            w = self.root.winfo_screenwidth()
            h = self.root.winfo_screenheight()
            self.root.geometry(f"{w}x{h}+0+0")
            
        # Discover videos
        self.video_extensions = ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.MP4", "*.AVI", "*.MKV", "*.MOV"]
        self.video_paths = []
        for ext in self.video_extensions:
            self.video_paths.extend(glob.glob(os.path.join(self.video_dir, ext)))
        self.video_paths = sorted(list(set(self.video_paths)))
        
        # App State
        self.current_video_idx = -1
        self.frames_gray = []       # Original resolution grayscale frames
        self.frames_rgb = []        # Original resolution RGB frames for display
        self.video_w = 0
        self.video_h = 0
        
        self.state = "idle"         # "idle", "tracking", "tracked", "preview"
        self.current_frame_idx = 0
        
        # Discrete tracking coordinates
        self.recorded_frames = []   # List of frame indices sampled (e.g. [120, 130, 140])
        self.recorded_coords = []   # List of (x, y) coordinates at original resolution matching the frames
        
        self.last_mouse_canvas_pos = (0, 0)
        self.playback_timer = None
        self.preview_idx = 0
        
        # Canvas display cache
        self.main_tk_image = None
        self.side_tk_image = None
        self.cache_preview_tk_image = None
        self.canvas_w = 100
        self.canvas_h = 100
        
        # Set default styling
        self.setup_styles()
        
        # Build UI layout
        self.build_ui()
        
        # Bind hotkeys
        self.root.bind("<Escape>", lambda e: self.toggle_fullscreen())
        self.root.bind("<space>", lambda e: self.on_space_pressed())
        
        # Load first video if any exist
        self.refresh_cache_list()
        if self.video_paths:
            self.select_video(0)
        else:
            self.update_status("No videos found in directory. Please specify a directory with videos.", "#ff3366")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background="#121212", foreground="#ffffff")
        style.configure("TLabel", background="#121212", foreground="#ffffff", font=("Helvetica", 10))
        style.configure("TFrame", background="#121212")
        style.configure("TButton", background="#2a2a2a", foreground="#ffffff", font=("Helvetica", 10, "bold"), borderwidth=0, focuscolor="none")
        style.map("TButton", background=[("active", "#3a3a3a"), ("disabled", "#1a1a1a")], foreground=[("disabled", "#555555")])
        style.configure("TCombobox", fieldbackground="#1e1e1e", background="#2d2d2d", foreground="#ffffff", darkcolor="#1e1e1e")

    def build_ui(self):
        # 1. Main Header
        header_frame = tk.Frame(self.root, bg="#1e1e1e", height=50)
        header_frame.pack(side="top", fill="x")
        header_frame.pack_propagate(False)
        
        title_lbl = tk.Label(header_frame, text="VIDEO DATASET GENERATOR V2 (DISCRETE)", font=("Helvetica", 13, "bold"), bg="#1e1e1e", fg="#00e6ff")
        title_lbl.pack(side="left", padx=15)
        
        self.fullscreen_lbl = tk.Label(header_frame, text="Press ESC to toggle Fullscreen", font=("Helvetica", 9, "italic"), bg="#1e1e1e", fg="#888888")
        self.fullscreen_lbl.pack(side="right", padx=15)
        
        # 2. Main Workspace Split
        workspace = tk.Frame(self.root, bg="#121212")
        workspace.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        
        # Left Panel (Video List & Config)
        left_panel = tk.Frame(workspace, bg="#1e1e1e", width=260, bd=1, relief="solid")
        left_panel.pack(side="left", fill="y", padx=5)
        left_panel.pack_propagate(False)
        
        list_lbl = tk.Label(left_panel, text="Videos List", font=("Helvetica", 11, "bold"), bg="#1e1e1e", fg="#00e6ff")
        list_lbl.pack(pady=(10, 2))
        
        # Scrollable Listbox (Videos)
        list_container = tk.Frame(left_panel, bg="#1e1e1e")
        list_container.pack(fill="both", expand=True, padx=10, pady=2)
        
        scrollbar = tk.Scrollbar(list_container, orient="vertical")
        self.listbox = tk.Listbox(list_container, bg="#121212", fg="#ffffff", selectbackground="#00e6ff", selectforeground="#121212", bd=0, highlightthickness=0, yscrollcommand=scrollbar.set, font=("Helvetica", 9))
        scrollbar.config(command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)
        
        # Populate Listbox
        for path in self.video_paths:
            self.listbox.insert(tk.END, os.path.basename(path))
            
        # Cache List Label
        cache_lbl = tk.Label(left_panel, text="Cache List", font=("Helvetica", 11, "bold"), bg="#1e1e1e", fg="#00e6ff")
        cache_lbl.pack(pady=(10, 2))
        
        # Scrollable Listbox (Cache)
        cache_container = tk.Frame(left_panel, bg="#1e1e1e")
        cache_container.pack(fill="both", expand=True, padx=10, pady=2)
        
        cache_scrollbar = tk.Scrollbar(cache_container, orient="vertical")
        self.cache_listbox = tk.Listbox(cache_container, bg="#121212", fg="#ffffff", selectbackground="#00e6ff", selectforeground="#121212", bd=0, highlightthickness=0, yscrollcommand=cache_scrollbar.set, font=("Helvetica", 9))
        cache_scrollbar.config(command=self.cache_listbox.yview)
        cache_scrollbar.pack(side="right", fill="y")
        self.cache_listbox.pack(side="left", fill="both", expand=True)
        self.cache_listbox.bind("<<ListboxSelect>>", self.on_cache_listbox_select)
            
        # Metadata / Configuration widgets in Left Panel
        config_frame = tk.LabelFrame(left_panel, text=" Target Settings ", bg="#1e1e1e", fg="#00e6ff", font=("Helvetica", 9, "bold"), padx=10, pady=10)
        config_frame.pack(fill="x", side="bottom", padx=10, pady=15)
        
        self.lbl_resolution = tk.Label(config_frame, text="Resolution: N/A", bg="#1e1e1e", fg="#cccccc", font=("Courier", 9), anchor="w")
        self.lbl_resolution.pack(fill="x", pady=2)
        
        self.lbl_frames = tk.Label(config_frame, text="Frames: N/A", bg="#1e1e1e", fg="#cccccc", font=("Courier", 9), anchor="w")
        self.lbl_frames.pack(fill="x", pady=2)
        
        # Step Size combobox
        step_lbl = tk.Label(config_frame, text="Frame Step Size:", bg="#1e1e1e", fg="#aaaaaa", font=("Helvetica", 9))
        step_lbl.pack(fill="x", pady=(5, 1), anchor="w")
        self.step_combo = ttk.Combobox(config_frame, values=["1", "2", "5", "10", "15", "20", "30"], state="readonly", width=12)
        self.step_combo.set("10")
        self.step_combo.pack(fill="x", pady=2)
        
        crop_lbl = tk.Label(config_frame, text="crop_max_size:", bg="#1e1e1e", fg="#aaaaaa", font=("Helvetica", 9))
        crop_lbl.pack(fill="x", pady=(5, 1), anchor="w")
        
        self.crop_combo = ttk.Combobox(config_frame, values=["128", "256", "512", "1024", "2048"], state="readonly", width=12)
        self.crop_combo.set("512")
        self.crop_combo.pack(fill="x", pady=2)
        
        # Right Panel (Previews)
        right_panel = tk.Frame(workspace, bg="#1e1e1e", width=240, bd=1, relief="solid")
        right_panel.pack(side="right", fill="y", padx=5)
        right_panel.pack_propagate(False)
        
        # 1. End Frame Preview of active video
        right_title = tk.Label(right_panel, text="Active Video: End Frame", font=("Helvetica", 10, "bold"), bg="#1e1e1e", fg="#ffffff")
        right_title.pack(pady=(10, 2))
        
        self.side_canvas = tk.Canvas(right_panel, bg="#121212", bd=0, highlightthickness=0, width=220, height=150)
        self.side_canvas.pack(padx=10, pady=2)
        
        # 2. Selected Cache Target Preview
        cache_preview_title = tk.Label(right_panel, text="Selected Cache Preview", font=("Helvetica", 10, "bold"), bg="#1e1e1e", fg="#ffffff")
        cache_preview_title.pack(pady=(15, 2))
        
        self.cache_canvas = tk.Canvas(right_panel, bg="#121212", bd=0, highlightthickness=0, width=220, height=150)
        self.cache_canvas.pack(padx=10, pady=2)
        
        self.lbl_cache_source = tk.Label(right_panel, text="Source: N/A", bg="#1e1e1e", fg="#aaaaaa", font=("Helvetica", 8, "italic"), anchor="w", justify="left", wrap=210)
        self.lbl_cache_source.pack(fill="x", padx=15, pady=(2, 10))
        
        help_desc = (
            "Instructions:\n"
            "1. Select a video from list.\n"
            "2. Scrub timeline below to find start.\n"
            "3. Click Start Tracking.\n"
            "4. Click target in main view. Video advances by Step.\n"
            "5. Click Stop Tracking when target leaves field of view."
        )
        help_lbl = tk.Label(right_panel, text=help_desc, font=("Helvetica", 8), bg="#1e1e1e", fg="#888888", justify="left", wrap=210)
        help_lbl.pack(fill="both", expand=True, padx=15, pady=2)
        
        # Center Frame
        center_panel = tk.Frame(workspace, bg="#121212")
        center_panel.pack(side="left", fill="both", expand=True, padx=5)
        
        # Main Canvas for frames
        self.main_canvas = tk.Canvas(center_panel, bg="#1a1a1a", bd=1, relief="solid", highlightthickness=0)
        self.main_canvas.pack(fill="both", expand=True, padx=5, pady=5)
        self.main_canvas.bind("<Configure>", self.on_canvas_resize)
        self.main_canvas.bind("<Button-1>", self.on_canvas_click)
        self.main_canvas.bind("<Motion>", self.on_mouse_move)
        
        # Video seek timeline slider
        self.seek_slider = tk.Scale(center_panel, from_=0, to=100, orient="horizontal", bg="#121212", fg="#00e6ff", highlightthickness=0, label="Video Timeline", font=("Helvetica", 8), command=self.on_seek_slider_change)
        self.seek_slider.pack(fill="x", padx=5, pady=2)
        
        # Bottom controls in center panel
        self.status_lbl = tk.Label(center_panel, text="Select a video to begin.", font=("Helvetica", 11, "bold"), bg="#121212", fg="#00ff66", height=2)
        self.status_lbl.pack(fill="x")
        
        controls_frame = tk.Frame(center_panel, bg="#1e1e1e", height=60, bd=1, relief="solid")
        controls_frame.pack(fill="x", pady=5)
        controls_frame.pack_propagate(False)
        
        # Buttons
        self.btn_start_stop = ttk.Button(controls_frame, text="▶ Start Tracking", state="disabled", command=self.toggle_tracking)
        self.btn_start_stop.pack(side="left", padx=15, pady=15)
        
        self.btn_save = ttk.Button(controls_frame, text="💾 Save to Cache", state="disabled", command=self.save_cache)
        self.btn_save.pack(side="left", padx=5)
        
        self.btn_preview = ttk.Button(controls_frame, text="▶ Play Preview", state="disabled", command=self.play_preview)
        self.btn_preview.pack(side="left", padx=5)
        
        self.btn_reset = ttk.Button(controls_frame, text="🔄 Start Over", state="disabled", command=self.reset_tracking)
        self.btn_reset.pack(side="left", padx=5)
        
        # Speed Scale for Preview Playback
        speed_container = tk.Frame(controls_frame, bg="#1e1e1e")
        speed_container.pack(side="right", padx=15, pady=5)
        
        self.speed_lbl = tk.Label(speed_container, text="Preview Delay: 150 ms", font=("Helvetica", 8), bg="#1e1e1e", fg="#cccccc")
        self.speed_lbl.pack(side="top")
        
        self.speed_slider = tk.Scale(speed_container, from_=40, to=300, orient="horizontal", length=150, bg="#1e1e1e", fg="#ffffff", highlightthickness=0, resolution=10, showvalue=False, command=self.on_speed_changed)
        self.speed_slider.set(150)
        self.speed_slider.pack(side="bottom")

    def toggle_fullscreen(self):
        state = not self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', state)

    def on_speed_changed(self, val):
        delay = int(val)
        self.speed_lbl.config(text=f"Preview Delay: {delay} ms")

    def update_status(self, text, color="#ffffff"):
        self.status_lbl.config(text=text, fg=color)

    def refresh_cache_list(self):
        self.cache_listbox.delete(0, tk.END)
        existing_flights = sorted(glob.glob(os.path.join(self.cache_dir, "flight_*.pkl")))
        for path in existing_flights:
            self.cache_listbox.insert(tk.END, os.path.basename(path))

    def on_listbox_select(self, event):
        selection = self.listbox.curselection()
        if selection:
            self.select_video(selection[0])

    def select_video(self, idx):
        if self.playback_timer:
            self.root.after_cancel(self.playback_timer)
            self.playback_timer = None
            
        self.current_video_idx = idx
        video_path = self.video_paths[idx]
        
        self.update_status(f"Loading {os.path.basename(video_path)}...", "#00e6ff")
        self.root.update_idletasks()
        
        # Load Video Frames
        cap = cv2.VideoCapture(video_path)
        self.frames_gray = []
        self.frames_rgb = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale (for dataset cache)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frames_gray.append(gray)
            
            # Convert to RGB (for Tkinter display)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames_rgb.append(rgb)
            
            frame_count += 1
            if frame_count > 1000:  # Allow longer videos for driving tracks (up to 1000 frames)
                print(f"[Warning] Video {os.path.basename(video_path)} truncated to 1000 frames to save RAM.")
                break
                
        cap.release()
        
        if not self.frames_rgb:
            self.update_status("Error loading video frames.", "#ff3366")
            return
            
        self.video_h, self.video_w = self.frames_gray[0].shape[:2]
        
        # Update metadata display
        self.lbl_resolution.config(text=f"Resolution: {self.video_w}x{self.video_h}")
        self.lbl_frames.config(text=f"Frames: {len(self.frames_rgb)}")
        
        # Configure seek bar slider
        self.seek_slider.config(to=len(self.frames_rgb) - 1)
        self.seek_slider.set(0)
        
        # Calculate default crop_max_size: closest power of 2 below minimum dimension
        min_dim = min(self.video_w, self.video_h)
        power = int(math.log2(min_dim))
        default_crop = 2**power
        
        if default_crop > 2048:
            default_crop = 2048
        elif default_crop < 128:
            default_crop = 128
            
        self.crop_combo.set(str(default_crop))
        
        # Reset State
        self.state = "idle"
        self.current_frame_idx = 0
        self.recorded_frames = []
        self.recorded_coords = []
        
        # Configure buttons
        self.btn_start_stop.config(text="▶ Start Tracking", state="normal")
        self.btn_reset.config(state="disabled")
        self.btn_save.config(state="disabled")
        self.btn_preview.config(state="disabled")
        
        # Redraw
        self.render_frames()
        self.update_status("Find start frame using timeline slider, then click 'Start Tracking'.", "#00e6ff")

    def on_seek_slider_change(self, val):
        self.current_frame_idx = int(val)
        self.render_frames()

    def on_canvas_resize(self, event):
        self.canvas_w = event.width
        self.canvas_h = event.height
        self.render_frames()

    def render_frames(self):
        if not self.frames_rgb:
            return
            
        # 1. Main Canvas Render
        frame = self.frames_rgb[self.current_frame_idx]
        h_f, w_f = frame.shape[:2]
        
        # Aspect Ratio Fit
        scale = min(self.canvas_w / w_f, self.canvas_h / h_f)
        w_disp = int(w_f * scale)
        h_disp = int(h_f * scale)
        
        if w_disp > 0 and h_disp > 0:
            pil_img = Image.fromarray(frame).resize((w_disp, h_disp), Image.Resampling.BILINEAR)
            draw = ImageDraw.Draw(pil_img)
            
            # Decorate frame based on state
            if self.state == "preview":
                # Render preview coordinate if current frame was sampled
                if self.current_frame_idx in self.recorded_frames:
                    idx = self.recorded_frames.index(self.current_frame_idx)
                    orig_target = self.recorded_coords[idx]
                    
                    tx_disp = int(orig_target[0] * scale)
                    ty_disp = int(orig_target[1] * scale)
                    
                    r = 15
                    draw.ellipse([tx_disp-r, ty_disp-r, tx_disp+r, ty_disp+r], outline="#00e6ff", width=2)
                    draw.ellipse([tx_disp-2, ty_disp-2, tx_disp+2, ty_disp+2], fill="#00e6ff")
                    
            elif self.state == "tracking":
                # Render already recorded keys that are visible
                for f_idx, coord in zip(self.recorded_frames, self.recorded_coords):
                    if f_idx == self.current_frame_idx:
                        tx_disp = int(coord[0] * scale)
                        ty_disp = int(coord[1] * scale)
                        
                        r = 15
                        draw.ellipse([tx_disp-r, ty_disp-r, tx_disp+r, ty_disp+r], outline="#00ff66", width=2)
                        draw.ellipse([tx_disp-2, ty_disp-2, tx_disp+2, ty_disp+2], fill="#00ff66")
                
                # Draw mouse crosshair feedback during tracking mode
                mx, my = self.last_mouse_canvas_pos
                x_offset = (self.canvas_w - w_disp) / 2
                y_offset = (self.canvas_h - h_disp) / 2
                
                mx_fit = mx - x_offset
                my_fit = my - y_offset
                
                r = 15
                draw.ellipse([mx_fit-r, my_fit-r, mx_fit+r, my_fit+r], outline="#ff3366", width=2)
                draw.ellipse([mx_fit-2, my_fit-2, mx_fit+2, my_fit+2], fill="#ff3366")
                
            elif self.state == "tracked":
                # Review mode: draw cyan circle if frame is in recorded frames list
                if self.current_frame_idx in self.recorded_frames:
                    idx = self.recorded_frames.index(self.current_frame_idx)
                    orig_target = self.recorded_coords[idx]
                    
                    tx_disp = int(orig_target[0] * scale)
                    ty_disp = int(orig_target[1] * scale)
                    
                    r = 15
                    draw.ellipse([tx_disp-r, ty_disp-r, tx_disp+r, ty_disp+r], outline="#00e6ff", width=2)
                    draw.ellipse([tx_disp-2, ty_disp-2, tx_disp+2, ty_disp+2], fill="#00e6ff")
            
            # Center on Canvas
            background = Image.new("RGB", (self.canvas_w, self.canvas_h), "#1a1a1a")
            offset_x = (self.canvas_w - w_disp) // 2
            offset_y = (self.canvas_h - h_disp) // 2
            background.paste(pil_img, (offset_x, offset_y))
            
            self.main_tk_image = ImageTk.PhotoImage(image=background)
            self.main_canvas.delete("all")
            self.main_canvas.create_image(0, 0, anchor="nw", image=self.main_tk_image)
            
        # 2. Side Canvas Render (End Frame)
        end_frame = self.frames_rgb[-1]
        h_e, w_e = end_frame.shape[:2]
        
        # Fit into side canvas
        side_w = self.side_canvas.winfo_width()
        side_h = self.side_canvas.winfo_height()
        if side_w < 10 or side_h < 10:
            side_w, side_h = 220, 150
            
        scale_e = min(side_w / w_e, side_h / h_e)
        w_side_disp = int(w_e * scale_e)
        h_side_disp = int(h_e * scale_e)
        
        if w_side_disp > 0 and h_side_disp > 0:
            pil_side = Image.fromarray(end_frame).resize((w_side_disp, h_side_disp), Image.Resampling.BILINEAR)
            background_side = Image.new("RGB", (side_w, side_h), "#121212")
            off_x = (side_w - w_side_disp) // 2
            off_y = (side_h - h_side_disp) // 2
            background_side.paste(pil_side, (off_x, off_y))
            
            self.side_tk_image = ImageTk.PhotoImage(image=background_side)
            self.side_canvas.delete("all")
            self.side_canvas.create_image(0, 0, anchor="nw", image=self.side_tk_image)

    def on_mouse_move(self, event):
        self.last_mouse_canvas_pos = (event.x, event.y)
        if self.state == "tracking":
            self.render_frames()

    def translate_coords_canvas_to_original(self, cx, cy):
        h_f, w_f = self.frames_gray[0].shape[:2]
        scale = min(self.canvas_w / w_f, self.canvas_h / h_f)
        w_disp = int(w_f * scale)
        h_disp = int(h_f * scale)
        
        x_offset = (self.canvas_w - w_disp) / 2
        y_offset = (self.canvas_h - h_disp) / 2
        
        fx = (cx - x_offset) / scale
        fy = (cy - y_offset) / scale
        
        fx = max(0, min(w_f - 1, int(fx)))
        fy = max(0, min(h_f - 1, int(fy)))
        
        return (fx, fy)

    def toggle_tracking(self):
        if self.state != "tracking":
            # Start tracking
            self.state = "tracking"
            self.recorded_frames = []
            self.recorded_coords = []
            
            self.btn_start_stop.config(text="■ Stop Tracking")
            self.btn_reset.config(state="disabled")
            self.btn_save.config(state="disabled")
            self.btn_preview.config(state="disabled")
            
            # Disable timeline seek bar and sidebar config
            self.seek_slider.config(state="disabled")
            self.listbox.config(state="disabled")
            self.step_combo.config(state="disabled")
            self.crop_combo.config(state="disabled")
            
            self.update_status(f"TRACKING ACTIVE: Click target. Video will step forward by {self.step_combo.get()}.", "#ff3366")
            self.render_frames()
        else:
            # Stop tracking
            self.stop_tracking()

    def stop_tracking(self):
        self.state = "tracked"
        self.btn_start_stop.config(text="▶ Start Tracking")
        
        # Re-enable controls
        self.seek_slider.config(state="normal")
        self.listbox.config(state="normal")
        self.step_combo.config(state="normal")
        self.crop_combo.config(state="normal")
        
        self.btn_reset.config(state="normal")
        
        if len(self.recorded_frames) >= 2:
            self.btn_save.config(state="normal")
            self.btn_preview.config(state="normal")
            self.update_status(f"Tracking stopped. {len(self.recorded_frames)} keyframes captured. Verify or Save.", "#00ff66")
        else:
            self.update_status("Tracking stopped. Need at least 2 keyframes to save.", "#ffaa00")
            
        self.render_frames()

    def on_canvas_click(self, event):
        if self.state == "tracking":
            # Click samples target coordinate
            target = self.translate_coords_canvas_to_original(event.x, event.y)
            self.recorded_frames.append(self.current_frame_idx)
            self.recorded_coords.append(target)
            
            step = int(self.step_combo.get())
            next_frame = self.current_frame_idx + step
            
            if next_frame >= len(self.frames_rgb):
                # reached end of video
                self.update_status("Reached end of video.", "#00ff66")
                self.stop_tracking()
            else:
                self.current_frame_idx = next_frame
                self.seek_slider.set(next_frame)
                self.render_frames()

    def play_preview(self):
        if self.state not in ["tracked", "preview"] or not self.recorded_frames:
            return
            
        if self.playback_timer:
            self.root.after_cancel(self.playback_timer)
            
        self.state = "preview"
        self.preview_idx = 0
        self.current_frame_idx = self.recorded_frames[0]
        self.seek_slider.set(self.current_frame_idx)
        self.update_status("PREVIEW: Playing back keyframes sequentially.", "#00e6ff")
        
        delay = self.speed_slider.get()
        self.playback_timer = self.root.after(delay, self.tick_preview)

    def tick_preview(self):
        if self.state != "preview":
            return
            
        self.preview_idx += 1
        if self.preview_idx >= len(self.recorded_frames):
            self.state = "tracked"
            self.current_frame_idx = self.recorded_frames[-1]
            self.seek_slider.set(self.current_frame_idx)
            self.update_status("Preview finished.", "#00ff66")
            self.render_frames()
        else:
            self.current_frame_idx = self.recorded_frames[self.preview_idx]
            self.seek_slider.set(self.current_frame_idx)
            self.render_frames()
            
            delay = self.speed_slider.get()
            self.playback_timer = self.root.after(delay, self.tick_preview)

    def on_space_pressed(self):
        if self.state == "preview":
            self.state = "tracked"
            self.update_status("Preview paused.", "#00ff66")
            self.render_frames()
        elif self.state == "tracked":
            self.play_preview()

    def reset_tracking(self):
        if self.playback_timer:
            self.root.after_cancel(self.playback_timer)
            self.playback_timer = None
            
        self.state = "idle"
        self.recorded_frames = []
        self.recorded_coords = []
        self.current_frame_idx = 0
        self.seek_slider.set(0)
        
        self.btn_save.config(state="disabled")
        self.btn_preview.config(state="disabled")
        self.btn_reset.config(state="disabled")
        
        self.render_frames()
        self.update_status("Tracking reset. Find start frame and click Start Tracking.", "#00e6ff")

    def save_cache(self):
        if self.state not in ["tracked", "preview"] or len(self.recorded_frames) < 2:
            messagebox.showerror("Error", "Not enough keyframes captured.")
            return
            
        # Discover next index
        existing_flights = glob.glob(os.path.join(self.cache_dir, "flight_*.pkl"))
        max_idx = -1
        for path in existing_flights:
            try:
                base = os.path.basename(path)
                num = int(base.split("_")[1].split(".")[0])
                if num > max_idx:
                    max_idx = num
            except Exception:
                pass
                
        next_idx = max_idx + 1
        output_filename = f"flight_{next_idx:04d}.pkl"
        output_path = os.path.join(self.cache_dir, output_filename)
        
        # Prepare data structure matching CARLA dataset compiler format
        # E.g. list of dictionaries
        flight_data = []
        num_saved_frames = len(self.recorded_frames)
        
        crop_max_sz = int(self.crop_combo.get())
        crop_min_sz = 4 
        
        for i in range(num_saved_frames):
            orig_frame_idx = self.recorded_frames[i]
            gray_img = self.frames_gray[orig_frame_idx]
            target_pt = self.recorded_coords[i]
            
            # Simulate a decaying distance to target (approaching behavior)
            # Starts at 50.0m and decays to 2.0m
            if num_saved_frames > 1:
                t = i / float(num_saved_frames - 1)
                dist = 50.0 - t * (50.0 - 2.0)
            else:
                dist = 2.0
                
            frame_dict = {
                # Map to sequential frame indices for the Siamese target compiler
                "frame_index": i,
                "image_gray": gray_img,
                "target_2d": target_pt,
                "target_3d": np.array([0.0, 0.0, 0.0]),
                "distance_to_target": dist
            }
            
            # Inject dynamic crop metadata in frame 0 for generic compiler
            if i == 0:
                frame_dict["crop_max_size"] = crop_max_sz
                frame_dict["crop_min_size"] = crop_min_sz
                frame_dict["video_name"] = os.path.basename(self.video_paths[self.current_video_idx])
                
            flight_data.append(frame_dict)
            
        # Save Pickle
        try:
            with open(output_path, "wb") as f:
                pickle.dump(flight_data, f)
                
            messagebox.showinfo("Success", f"Flight saved successfully as {output_filename}!")
            
            # Refresh Cache List Box
            self.refresh_cache_list()
            
            # Reset tracking on the current video
            self.reset_tracking()
            
            # Show saved message
            self.update_status(f"Saved successfully as {output_filename}. Ready for another tracking on current video.", "#00ff66")
        except Exception as e:
            messagebox.showerror("Error Saving", f"Failed to save pickle file: {e}")

    def on_cache_listbox_select(self, event):
        selection = self.cache_listbox.curselection()
        if not selection:
            return
            
        filename = self.cache_listbox.get(selection[0])
        filepath = os.path.join(self.cache_dir, filename)
        
        try:
            with open(filepath, 'rb') as f:
                flight_data = pickle.load(f)
                
            if not flight_data:
                return
                
            frame_0 = flight_data[0]
            gray_img = frame_0["image_gray"]
            target_pt = frame_0["target_2d"]
            video_name = frame_0.get("video_name", "N/A")
            
            self.lbl_cache_source.config(text=f"Source: {video_name}")
            
            # Convert grayscale to RGB for PIL drawing
            rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(rgb_img)
            
            # Draw target circle
            draw = ImageDraw.Draw(pil_img)
            tx, ty = int(target_pt[0]), int(target_pt[1])
            r = max(10, int(gray_img.shape[1] * 0.015))
            draw.ellipse([tx-r, ty-r, tx+r, ty+r], outline="#ff3366", width=max(2, int(r*0.15)))
            draw.ellipse([tx-4, ty-4, tx+4, ty+4], fill="#ff3366")
            
            # Resize image to fit cache_canvas
            canvas_w = self.cache_canvas.winfo_width()
            canvas_h = self.cache_canvas.winfo_height()
            if canvas_w < 10 or canvas_h < 10:
                canvas_w, canvas_h = 220, 150
                
            h_f, w_f = gray_img.shape[:2]
            scale = min(canvas_w / w_f, canvas_h / h_f)
            w_disp = int(w_f * scale)
            h_disp = int(h_f * scale)
            
            if w_disp > 0 and h_disp > 0:
                pil_resized = pil_img.resize((w_disp, h_disp), Image.Resampling.BILINEAR)
                background = Image.new("RGB", (canvas_w, canvas_h), "#121212")
                off_x = (canvas_w - w_disp) // 2
                off_y = (canvas_h - h_disp) // 2
                background.paste(pil_resized, (off_x, off_y))
                
                self.cache_preview_tk_image = ImageTk.PhotoImage(image=background)
                self.cache_canvas.delete("all")
                self.cache_canvas.create_image(0, 0, anchor="nw", image=self.cache_preview_tk_image)
        except Exception as e:
            print(f"[Error] Failed to load preview for {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="TargetTrackerVer4 Video Dataset Manual Tracking Tool V2")
    parser.add_argument("--video_dir", default="videos4dataset", help="Directory containing input videos")
    parser.add_argument("--cache_dir", default="cache", help="Directory to save generated flight PKLs")
    args = parser.parse_args()
    
    # Verify video directory exists
    if not os.path.exists(args.video_dir):
        print(f"Error: Video directory '{args.video_dir}' does not exist.")
        # If it doesn't exist, search parent folders or try tracker/videos4dataset
        fallback = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "videos4dataset"))
        if os.path.exists(fallback):
            print(f"Using fallback video directory: {fallback}")
            args.video_dir = fallback
        else:
            os.makedirs(args.video_dir, exist_ok=True)
            print(f"Created empty video directory: {args.video_dir}. Please drop MP4 videos inside.")
            
    root = tk.Tk()
    app = VideoDatasetGeneratorV2(root, args.video_dir, args.cache_dir)
    root.mainloop()

if __name__ == "__main__":
    main()
