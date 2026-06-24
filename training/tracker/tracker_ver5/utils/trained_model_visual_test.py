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

class ModelInferenceVisualizer:
    def __init__(self, root, dataset_dir, model_path, threshold=0.5, iterations_num=1):
        # Import tracker_model to register custom layers
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(script_dir)
        sys.path.append(os.path.dirname(script_dir))
        import tracker_model
        
        import tensorflow as tf
        
        self.threshold = threshold
        self.model_path = model_path
        self.iterations_num = iterations_num
        
        # Instance variables to cache current frame prediction and metadata
        self.curr_ref_stack = None
        self.curr_search_raw = None
        self.curr_gt_coords = None
        self.curr_gt_quality = None
        self.curr_meta = None
        self.curr_pred_heatmap = None
        self.curr_pred_quality = None
        self.curr_search_256 = None
        self.is_tflite = model_path.endswith('.tflite')
        
        if self.is_tflite:
            print(f"Loading TFLite model from {model_path}...")
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Map input tensors
            self.ref_input_idx = None
            self.search_input_idx = None
            for detail in self.input_details:
                name = detail['name'].lower()
                if 'reference_stack' in name:
                    self.ref_input_idx = detail['index']
                elif 'search_frame' in name:
                    self.search_input_idx = detail['index']
            
            if self.ref_input_idx is None or self.search_input_idx is None:
                # Fallback to shape matching
                for detail in self.input_details:
                    shape = list(detail['shape'])
                    if 16 in shape or 64 in shape:
                        self.ref_input_idx = detail['index']
                    else:
                        self.search_input_idx = detail['index']
                        
            # Map output tensors
            self.heatmap_output_idx = None
            self.quality_output_idx = None
            for detail in self.output_details:
                shape = list(detail['shape'])
                # Map based on shape: heatmap has 4 dimensions [1, H, W, 1] with H, W > 1
                if len(shape) == 4 and shape[1] > 1 and shape[2] > 1:
                    self.heatmap_output_idx = detail['index']
                else:
                    self.quality_output_idx = detail['index']
            print(f"Mapped inputs: ref={self.ref_input_idx}, search={self.search_input_idx}")
            print(f"Mapped outputs: heatmap={self.heatmap_output_idx}, quality={self.quality_output_idx}")
        else:
            print(f"Loading Keras model from {model_path}...")
            custom_objects = {
                "DepthwiseCorrelationFusion": tracker_model.DepthwiseCorrelationFusion,
                "Conv2DCorrelationFusion": tracker_model.Conv2DCorrelationFusion,
                "DepthToSpace": tracker_model.DepthToSpace,
            }
            is_qat = "qat" in os.path.basename(model_path).lower()
            if is_qat:
                try:
                    import tensorflow_model_optimization as tfmot

                    @tf.keras.utils.register_keras_serializable(package="Custom")
                    class CustomLayerQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
                        def get_weights_and_quantizers(self, layer):
                            return []
                        def get_activations_and_quantizers(self, layer):
                            return []
                        def set_quantize_weights(self, layer, quantize_weights):
                            pass
                        def set_quantize_activations(self, layer, quantize_activations):
                            pass
                        def get_output_quantizers(self, layer):
                            return [
                                tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
                                    num_bits=8, per_axis=False, symmetric=False, narrow_range=False
                                )
                            ]
                        def get_config(self):
                            return {}

                    custom_objects["CustomLayerQuantizeConfig"] = CustomLayerQuantizeConfig

                    # Monkeypatch QuantizeWrapperV2.build to build the inner layer first
                    orig_build = tfmot.quantization.keras.QuantizeWrapperV2.build
                    def custom_build(self, input_shape):
                        if not self.layer.built:
                            self.layer.build(input_shape)
                        orig_build(self, input_shape)
                    tfmot.quantization.keras.QuantizeWrapperV2.build = custom_build

                    with tfmot.quantization.keras.quantize_scope(custom_objects):
                        self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
                except ImportError:
                    print("Warning: tensorflow_model_optimization not installed. Trying normal load.")
                    self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False, custom_objects=custom_objects)
            else:
                self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False, custom_objects=custom_objects)
            self.model.summary()
        
        self.detect_model_shapes()
        
        self.dataset_dir = dataset_dir
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{self.dataset_dir}' does not exist.")
            
        h5_path = os.path.join(self.dataset_dir, "dataset.h5")
        if os.path.exists(h5_path):
            import h5py
            self.is_h5 = True
            with h5py.File(h5_path, 'r') as f:
                self.h5_size = f['reference_stack'].shape[0]
            print(f"Discovered HDF5 dataset dataset.h5 with {self.h5_size} samples.")
        else:
            self.is_h5 = False
            self.pickle_files = [f for f in os.listdir(self.dataset_dir) if f.endswith('.pkl')]
            if not self.pickle_files:
                raise FileNotFoundError(f"No dataset.h5 or pickle files found in '{self.dataset_dir}'.")
            self.pickle_files.sort()
            print(f"Discovered {len(self.pickle_files)} dataset pickle files.")
            
        self.current_pickle_idx = 0
        self.current_sample_idx = -1
        self.current_batch_data = None
        self.current_batch_size = 0
        
        self.root = root
        self.root.title("TargetTrackerVer4 - Live Inference Visualizer")
        self.root.geometry("880x500")
        self.root.configure(bg="#121212")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.setup_ui()
        
        self.root.bind("<space>", lambda e: self.load_next_sample())
        self.root.bind("<Escape>", lambda e: self.on_close())
        
        self.load_next_sample()

    def detect_model_shapes(self):
        self.search_frame_size = None
        self.stack_target_size = None
        self.stack_layers = None

        if self.is_tflite:
            try:
                for detail in self.input_details:
                    name = detail['name'].lower()
                    shape = list(detail['shape'])
                    if 'reference_stack' in name or 16 in shape or 64 in shape:
                        if shape[1] in (1, 2):  # NCHW
                            self.stack_layers = shape[1]
                            self.stack_target_size = shape[2]
                        else:  # NHWC
                            self.stack_target_size = shape[1]
                            self.stack_layers = shape[3]
                    elif 'search_frame' in name or len(shape) == 4:
                        if shape[1] == 1:  # NCHW
                            self.search_frame_size = shape[2]
                        else:  # NHWC
                            self.search_frame_size = shape[1]
                print(f"--> Inspected TFLite model shapes: search_frame_size={self.search_frame_size}, stack_target_size={self.stack_target_size}, stack_layers={self.stack_layers}")
            except Exception as e:
                print(f"--> Could not inspect TFLite model shapes directly: {e}")
        else:
            try:
                ref_idx = 0
                search_idx = 1
                if len(self.model.inputs) >= 2:
                    if 'search' in self.model.inputs[0].name.lower():
                        ref_idx = 1
                        search_idx = 0
                    ref_shape = self.model.inputs[ref_idx].shape.as_list()
                    search_shape = self.model.inputs[search_idx].shape.as_list()
                    self.stack_target_size = ref_shape[1]
                    self.stack_layers = ref_shape[3]
                    self.search_frame_size = search_shape[1]
                    print(f"--> Inspected Keras model shapes: search_frame_size={self.search_frame_size}, stack_target_size={self.stack_target_size}, stack_layers={self.stack_layers}")
            except Exception as e:
                print(f"--> Could not inspect Keras model shapes directly: {e}")

        # Fallback to model.conf
        if self.search_frame_size is None or self.stack_target_size is None:
            try:
                import configparser
                config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model.conf")
                config = configparser.ConfigParser()
                config.read(config_path)
                self.stack_layers = config.getint("Stack", "stack_layers", fallback=2)
                self.stack_target_size = config.getint("Stack", "stack_target_size", fallback=128)
                self.search_frame_size = config.getint("Stack", "search_frame_size", fallback=256)
                print(f"--> Loaded config from {config_path} as fallback: search_frame_size={self.search_frame_size}, stack_target_size={self.stack_target_size}, stack_layers={self.stack_layers}")
            except Exception as e:
                print(f"--> Could not read model.conf fallback: {e}")
                self.stack_layers = 2
                self.stack_target_size = 128
                self.search_frame_size = 256

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
        backend_name = "TFLite" if self.is_tflite else "Keras"
        header = tk.Label(self.root, text=f"TargetTrackerVer4 Visualizer ({backend_name} Backend)", font=("Outfit", 15, "bold"), bg="#121212", fg="#ffffff")
        header.pack(pady=15)
        
        # Control panel for algorithm selection
        self.control_frame = tk.Frame(self.root, bg="#121212")
        self.control_frame.pack(pady=5)
        
        tk.Label(self.control_frame, text="Centroid Algorithm:", font=("Inter", 10, "bold"), bg="#121212", fg="#ffffff").pack(side="left", padx=5)
        
        self.algo_var = tk.StringVar(value="Argmax (Discrete)")
        self.algo_options = [
            "Argmax (Discrete)",
            "Sub-pixel (Otsu)",
            "Sub-pixel (Triangle)",
            "Sub-pixel (Energy 75%)",
            "Sub-pixel (Mean + 1.5*Std)"
        ]
        self.algo_dropdown = tk.OptionMenu(
            self.control_frame, 
            self.algo_var, 
            *self.algo_options,
            command=self.on_algo_changed
        )
        self.algo_dropdown.config(bg="#1a1a1a", fg="#ffffff", activebackground="#333333", activeforeground="#ffffff", highlightthickness=0, bd=1)
        self.algo_dropdown["menu"].config(bg="#1a1a1a", fg="#ffffff")
        self.algo_dropdown.pack(side="left", padx=5)
        
        # Decide layout based on iterations_num
        if self.iterations_num <= 2:
            self.root.geometry(f"{290 * (2 + self.iterations_num) + 40}x520")
            self.frames_frame = tk.Frame(self.root, bg="#121212")
            self.frames_frame.pack(pady=10)
            
            self.ref_panel, self.ref_lbl = self.create_frame_slot(self.frames_frame, "Reference Features")
            self.search_panel, self.search_lbl = self.create_frame_slot(self.frames_frame, "Search Frame")
            
            self.heatmap_panels = []
            self.heatmap_lbls = []
            for i in range(self.iterations_num):
                p, l = self.create_frame_slot(self.frames_frame, f"Heatmap (Iter {i})")
                self.heatmap_panels.append(p)
                self.heatmap_lbls.append(l)
        else:
            width = max(3, self.iterations_num) * 290 + 40
            self.root.geometry(f"{width}x820")
            
            top_row = tk.Frame(self.root, bg="#121212")
            top_row.pack(pady=10)
            self.ref_panel, self.ref_lbl = self.create_frame_slot(top_row, "Reference Features")
            self.search_panel, self.search_lbl = self.create_frame_slot(top_row, "Search Frame")
            
            bottom_row = tk.Frame(self.root, bg="#121212")
            bottom_row.pack(pady=10)
            
            self.heatmap_panels = []
            self.heatmap_lbls = []
            for i in range(self.iterations_num):
                p, l = self.create_frame_slot(bottom_row, f"Heatmap (Iter {i})")
                self.heatmap_panels.append(p)
                self.heatmap_lbls.append(l)
                
        self.status_bar = tk.Label(self.root, text="Press [Space] for Next Sample", font=("Inter", 10), bg="#1c1c1c", fg="#aaaaaa", anchor="w", padx=15, pady=8)
        self.status_bar.pack(side="bottom", fill="x")
        
    def on_algo_changed(self, value):
        if self.curr_pred_heatmap is not None:
            try:
                self.update_visualization()
            except Exception as e:
                self.status_bar.config(text=f"Error updating: {e}")
        
    def load_next_sample(self):
        try:
            import tensorflow as tf
            
            if self.is_h5:
                self.current_sample_idx = (self.current_sample_idx + 1) % self.h5_size
                import h5py
                with h5py.File(os.path.join(self.dataset_dir, "dataset.h5"), 'r') as f:
                    ref_stack = f["reference_stack"][self.current_sample_idx]  # shape: (64, 64, 16)
                    search_raw = f["search_frame"][self.current_sample_idx]    # shape: (256, 256, 1)
                    gt_coords = f["ground_truth_coords"][self.current_sample_idx]  # shape: (2,)
                    gt_quality = f["ground_truth_quality"][self.current_sample_idx][0]
                
                meta = {
                    "flight_id": "dataset.h5",
                    "frame_idx": self.current_sample_idx,
                    "distance": 0.0,
                    "quality": gt_quality
                }
            else:
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
                ref_stack = sample["reference_stack"].transpose(3, 1, 2, 0)
                search_raw = sample["search_frame"]
                gt_coords = sample["ground_truth_coords"]
                meta = sample["metadata"]
                gt_quality = meta.get("quality", 1.0)
            
            # Save raw values to instance variables
            self.curr_ref_stack = ref_stack
            self.curr_search_raw = search_raw
            self.curr_gt_coords = gt_coords
            self.curr_gt_quality = gt_quality
            self.curr_meta = meta
            
            search_256 = cv2.resize(search_raw[:, :, 0], (256, 256), interpolation=cv2.INTER_LINEAR)
            self.curr_search_256 = search_256
            self.curr_pred_heatmap = True
            
            self.update_visualization()
            
        except Exception as e:
            self.status_bar.config(text=f"Error: {e}")
            import traceback
            traceback.print_exc()

    def compute_otsu_threshold(self, window):
        flat = window.ravel()
        if len(flat) == 0:
            return 0.0
        vmin, vmax = np.min(flat), np.max(flat)
        if vmin == vmax:
            return vmin
        hist, bin_edges = np.histogram(flat, bins=256, range=(vmin, vmax))
        bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        total = len(flat)
        sum_total = np.sum(bin_mids * hist)
        sum_b = 0.0
        w_b = 0.0
        max_variance = 0.0
        threshold = vmin
        for i in range(256):
            w_b += hist[i]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += bin_mids[i] * hist[i]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            variance = w_b * w_f * (m_b - m_f) ** 2
            if variance > max_variance:
                max_variance = variance
                threshold = bin_mids[i]
        return threshold

    def compute_triangle_threshold(self, window):
        flat = window.ravel()
        if len(flat) == 0:
            return 0.0
        vmin, vmax = np.min(flat), np.max(flat)
        if vmin == vmax:
            return vmin
        hist, bin_edges = np.histogram(flat, bins=256, range=(vmin, vmax))
        bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        hist_peak_idx = np.argmax(hist)
        if hist_peak_idx < 128:
            start_idx = hist_peak_idx
            end_idx = 255
            step = 1
        else:
            start_idx = hist_peak_idx
            end_idx = 0
            step = -1
        p1 = np.array([start_idx, hist[start_idx]])
        p2 = np.array([end_idx, hist[end_idx]])
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return bin_mids[hist_peak_idx]
        max_dist = -1.0
        best_idx = start_idx
        for i in range(start_idx, end_idx + step, step):
            p = np.array([i, hist[i]])
            dist = np.abs(np.cross(p2 - p1, p1 - p)) / line_len
            if dist > max_dist:
                max_dist = dist
                best_idx = i
        return bin_mids[best_idx]

    def compute_subpixel_position(self, heatmap, y_max, x_max, method):
        h_h, w_h = heatmap.shape
        window_size = 15
        half = window_size // 2
        y_start = max(0, y_max - half)
        y_end = min(h_h, y_max + half + 1)
        x_start = max(0, x_max - half)
        x_end = min(w_h, x_max + half + 1)
        
        window = heatmap[y_start:y_end, x_start:x_end]
        
        if method == "Sub-pixel (Otsu)":
            thresh = self.compute_otsu_threshold(window)
        elif method == "Sub-pixel (Triangle)":
            thresh = self.compute_triangle_threshold(window)
        elif method == "Sub-pixel (Mean + 1.5*Std)":
            mean_val = np.mean(window)
            std_val = np.std(window)
            thresh = mean_val + 1.5 * std_val
        elif method == "Sub-pixel (Energy 75%)":
            h_w, w_w = window.shape
            coords_with_val = []
            for r in range(h_w):
                for c in range(w_w):
                    coords_with_val.append((r + y_start, c + x_start, float(window[r, c])))
            coords_with_val.sort(key=lambda item: item[2], reverse=True)
            total_sum = sum(item[2] for item in coords_with_val)
            if total_sum == 0:
                return float(y_max), float(x_max)
            
            accum = 0.0
            selected_pixels = []
            for cy, cx, val in coords_with_val:
                selected_pixels.append((cy, cx))
                accum += val
                if accum >= 0.75 * total_sum:
                    break
            
            sum_w = 0.0
            sum_wy = 0.0
            sum_wx = 0.0
            for cy, cx in selected_pixels:
                w = float(heatmap[cy, cx])
                sum_w += w
                sum_wy += cy * w
                sum_wx += cx * w
            if sum_w > 0:
                return sum_wy / sum_w, sum_wx / sum_w
            return float(y_max), float(x_max)
        else:
            return float(y_max), float(x_max)
            
        local_y = y_max - y_start
        local_x = x_max - x_start
        
        visited = np.zeros_like(window, dtype=bool)
        queue = [(int(local_y), int(local_x))]
        visited[local_y, local_x] = True
        
        blob_pixels = []
        head = 0
        h_w, w_w = window.shape
        while head < len(queue):
            cy, cx = queue[head]
            head += 1
            blob_pixels.append((cy, cx))
            
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h_w and 0 <= nx < w_w:
                        if not visited[ny, nx] and window[ny, nx] > thresh:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                            
        sum_w = 0.0
        sum_wy = 0.0
        sum_wx = 0.0
        for cy, cx in blob_pixels:
            global_y = cy + y_start
            global_x = cx + x_start
            w = float(heatmap[global_y, global_x])
            sum_w += w
            sum_wy += global_y * w
            sum_wx += global_x * w
            
        if sum_w > 0:
            return sum_wy / sum_w, sum_wx / sum_w
        return float(y_max), float(x_max)

    def crop_and_resize(self, img, cx, cy, crop_size, target_size):
        h, w = img.shape[:2]
        half = crop_size / 2.0
        
        x1 = int(round(cx - half))
        x2 = int(round(cx + half))
        y1 = int(round(cy - half))
        y2 = int(round(cy + half))
        
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        crop = img[y1:y2, x1:x2].copy()
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
            
        return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    def crop_and_resize_ref_stack(self, ref_stack, crop_size, target_size):
        is_4d = (ref_stack.ndim == 4)
        working_stack = ref_stack[0] if is_4d else ref_stack
        h, w, c = working_stack.shape
        cx, cy = w / 2.0, h / 2.0
        
        layers_list = []
        for i in range(c):
            layer = working_stack[:, :, i]
            resized = self.crop_and_resize(layer, cx, cy, crop_size, target_size)
            layers_list.append(resized)
        cropped_stack = np.stack(layers_list, axis=-1)
        
        if is_4d:
            return np.expand_dims(cropped_stack, axis=0)
        return cropped_stack

    def map_to_original(self, x, y, crop_history):
        curr_x = x
        curr_y = y
        for cx, cy, crop_size in reversed(crop_history):
            tl_x = cx - crop_size / 2.0
            tl_y = cy - crop_size / 2.0
            curr_x = tl_x + curr_x * (crop_size / self.search_frame_size)
            curr_y = tl_y + curr_y * (crop_size / self.search_frame_size)
        return curr_x, curr_y

    def update_visualization(self):
        import tensorflow as tf
        
        ref_stack = self.curr_ref_stack
        search_raw = self.curr_search_raw
        gt_coords = self.curr_gt_coords
        gt_quality = self.curr_gt_quality
        meta = self.curr_meta
        
        method = self.algo_var.get()
        
        # Prepare lists to hold results for each iteration
        iter_searches = []
        iter_heatmaps = []
        iter_pred_qualities = []
        crop_history = []  # List of (cx, cy, crop_size)
        iter_preds_in_iter_space = [] # List of (x, y)
        
        # Initialize search frame and reference stack
        curr_search = cv2.resize(search_raw[:, :, 0], (self.search_frame_size, self.search_frame_size), interpolation=cv2.INTER_LINEAR)
        curr_ref = ref_stack.copy()
        
        # Resize reference if shapes do not match
        is_4d = (curr_ref.ndim == 4)
        working_stack = curr_ref[0] if is_4d else curr_ref
        ref_h, ref_w = working_stack.shape[:2]
        if ref_h != self.stack_target_size or ref_w != self.stack_target_size:
            curr_ref = self.crop_and_resize_ref_stack(curr_ref, ref_w, self.stack_target_size)
            ref_w = self.stack_target_size
        
        for iter_idx in range(self.iterations_num):
            # Prepare inputs for model
            if curr_ref.ndim == 4 and curr_ref.shape[0] == 1:
                ref_tensor = tf.cast(curr_ref, tf.float32)
            else:
                ref_tensor = tf.expand_dims(tf.cast(curr_ref, tf.float32), 0)
                
            if tf.reduce_max(ref_tensor) > 1.001:
                ref_tensor = ref_tensor / 255.0
            
            if curr_search.dtype == np.uint8:
                search_float = curr_search.astype(np.float32) / 255.0
            else:
                search_float = curr_search.astype(np.float32)
            search_tensor = tf.expand_dims(tf.expand_dims(search_float, -1), 0)
            
            # Predict
            if self.is_tflite:
                ref_array = ref_tensor.numpy() if hasattr(ref_tensor, "numpy") else np.array(ref_tensor)
                search_array = search_tensor.numpy() if hasattr(search_tensor, "numpy") else np.array(search_tensor)
                
                self.interpreter.set_tensor(self.ref_input_idx, ref_array)
                self.interpreter.set_tensor(self.search_input_idx, search_array)
                self.interpreter.invoke()
                
                pred_heatmap = self.interpreter.get_tensor(self.heatmap_output_idx)[0]
                if self.quality_output_idx is not None:
                    pred_quality = self.interpreter.get_tensor(self.quality_output_idx)[0][0]
                else:
                    pred_quality = 1.0
            else:
                pred = self.model([ref_tensor, search_tensor], training=False)
                if isinstance(pred, list) or isinstance(pred, tuple):
                    pred_heatmap = pred[0].numpy()[0]  # (H, W, 1)
                    pred_quality = pred[1].numpy()[0][0]  # scalar float
                else:
                    pred_heatmap = pred.numpy()[0]
                    pred_quality = 1.0
            
            heatmap = pred_heatmap[:, :, 0]
            iter_searches.append(curr_search.copy())
            iter_heatmaps.append(heatmap.copy())
            iter_pred_qualities.append(pred_quality)
            
            # Find peak coordinates using algorithm selected in GUI
            flat_idx = np.argmax(heatmap)
            y_max, x_max = np.unravel_index(flat_idx, heatmap.shape)
            if method == "Argmax (Discrete)":
                y_pred, x_pred = float(y_max), float(x_max)
            else:
                y_pred, x_pred = self.compute_subpixel_position(heatmap, y_max, x_max, method)
            
            # Scale peak coordinates to iteration space of the current iteration
            pcx = (x_pred / heatmap.shape[1]) * self.search_frame_size
            pcy = (y_pred / heatmap.shape[0]) * self.search_frame_size
            iter_preds_in_iter_space.append((pcx, pcy))
            
            if iter_idx < self.iterations_num - 1:
                # Save the crop history: center (pcx, pcy) and crop size
                crop_size = self.search_frame_size / 2.0
                crop_history.append((pcx, pcy, crop_size))
                
                # Crop and resize search frame for the next iteration
                curr_search = self.crop_and_resize(curr_search, pcx, pcy, crop_size, self.search_frame_size)
                
                # Crop and resize reference stack for the next iteration (crop size = ref_w / 2)
                curr_ref = self.crop_and_resize_ref_stack(curr_ref, ref_w / 2, ref_w)
                
        # Now map all predicted coordinates back to the original space (which is self.search_frame_size)
        iter_preds_mapped_search_space = []
        for idx in range(self.iterations_num):
            px, py = iter_preds_in_iter_space[idx]
            mx, my = self.map_to_original(px, py, crop_history[:idx])
            iter_preds_mapped_search_space.append((mx, my))
            
        # Map to display space (256x256) for drawing
        iter_preds_mapped = []
        for mx, my in iter_preds_mapped_search_space:
            dmx = mx * (256.0 / self.search_frame_size)
            dmy = my * (256.0 / self.search_frame_size)
            iter_preds_mapped.append((dmx, dmy))
            
        final_pred = iter_preds_mapped[-1]
        
        # Prepare GT label
        if gt_quality > 0.5:
            norm_x = gt_coords[1]
            norm_y = gt_coords[0]
            norm_coords = [norm_x, norm_y]
            curr_lbl_text = f"GT: [{norm_x:.2f}, {norm_y:.2f}]"
            curr_lbl_fg = "#00e6ff"
        else:
            norm_coords = None
            curr_lbl_text = "GT: None (Negative)"
            curr_lbl_fg = "#ff3366"
            
        # Draw Reference (just show first layer of the starting stack)
        if ref_stack.ndim == 3:
            ref_layer_0 = ref_stack[:, :, 0]
        elif ref_stack.ndim == 4 and ref_stack.shape[0] == 1:
            ref_layer_0 = ref_stack[0, :, :, 0]
        else:
            ref_layer_0 = ref_stack[0, :, :, 0]
            
        if ref_layer_0.dtype != np.uint8:
            ref_layer_0 = (ref_layer_0 * 255.0).astype(np.uint8)
        ref_vis = cv2.resize(ref_layer_0, (256, 256), interpolation=cv2.INTER_NEAREST)
        self.tk_img_ref = ImageTk.PhotoImage(Image.fromarray(ref_vis))
        self.ref_panel.config(image=self.tk_img_ref)
        self.ref_lbl.config(text="Target Features")
        
        # Draw Search Frame (original search frame with GT and final + intermediate predictions)
        search_rgb = cv2.cvtColor(self.curr_search_256, cv2.COLOR_GRAY2RGB)
        search_vis = (search_rgb.copy() * 255).astype(np.uint8)
        
        # Draw GT
        if norm_coords is not None:
            cx = int(norm_coords[0] * 256.0)
            cy = int(norm_coords[1] * 256.0)
            cv2.circle(search_vis, (cx, cy), 6, (0, 230, 255), 2)
            cv2.circle(search_vis, (cx, cy), 2, (0, 230, 255), -1)
            
        # Draw Intermediate predictions (as small red circles and yellow lines connecting them)
        for idx in range(self.iterations_num - 1):
            mx, my = iter_preds_mapped[idx]
            imx, imy = int(mx), int(my)
            cv2.circle(search_vis, (imx, imy), 3, (0, 0, 255), -1) # Red for intermediate
            if idx > 0:
                prev_mx, prev_my = iter_preds_mapped[idx - 1]
                cv2.line(search_vis, (int(prev_mx), int(prev_my)), (imx, imy), (0, 255, 255), 1) # Yellow line
                
        if self.iterations_num > 1:
            # Draw line to final prediction
            prev_mx, prev_my = iter_preds_mapped[-2]
            cv2.line(search_vis, (int(prev_mx), int(prev_my)), (int(final_pred[0]), int(final_pred[1])), (0, 255, 255), 1)
            
        # Draw Final prediction (as green circle)
        pcx, pcy = int(final_pred[0]), int(final_pred[1])
        cv2.circle(search_vis, (pcx, pcy), 6, (51, 255, 51), 2)
        cv2.circle(search_vis, (pcx, pcy), 2, (51, 255, 51), -1)
        
        self.tk_img_search = ImageTk.PhotoImage(Image.fromarray(search_vis))
        self.search_panel.config(image=self.tk_img_search)
        self.search_lbl.config(text=curr_lbl_text, fg=curr_lbl_fg)
        
        # Draw all predicted heatmaps
        self.tk_imgs_predicted = []
        for idx in range(self.iterations_num):
            heatmap = iter_heatmaps[idx]
            heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            tk_img = ImageTk.PhotoImage(Image.fromarray(heatmap_color_rgb))
            self.tk_imgs_predicted.append(tk_img)
            
            panel = self.heatmap_panels[idx]
            lbl = self.heatmap_lbls[idx]
            
            panel.config(image=tk_img)
            
            # Show prediction info for this iteration
            mx, my = iter_preds_mapped[idx]
            px, py = iter_preds_in_iter_space[idx]
            
            if norm_coords is not None:
                error = np.sqrt((mx/256.0 - norm_coords[0])**2 + (my/256.0 - norm_coords[1])**2) * 256.0
                error_str = f"Error: {error:.1f}px"
            else:
                error_str = "Error: N/A"
                
            info_text = f"Mapped: [{mx:.1f}, {my:.1f}]\nIter: [{px:.1f}, {py:.1f}]\n{error_str}\nQuality: {iter_pred_qualities[idx]:.2f}"
            lbl.config(text=info_text, fg="#33ff33")
            
        # Update status bar
        self.status_bar.config(text=f"Flight: {meta['flight_id']} | Frame: {meta['frame_idx']} | Dist: {meta['distance']:.1f}m | Press Space")

    def on_close(self):
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="dataset/")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--threshold", type=float, default=0.5, help="Heatmap noise threshold")
    parser.add_argument("--iterations_num", type=int, default=1, help="Number of refinement iterations (1 = standard single-pass)")
    args = parser.parse_args()
    
    root = tk.Tk()
    app = ModelInferenceVisualizer(
        root, 
        args.dataset_dir, 
        args.model_path, 
        threshold=args.threshold,
        iterations_num=args.iterations_num
    )
    root.mainloop()

if __name__ == "__main__":
    main()
