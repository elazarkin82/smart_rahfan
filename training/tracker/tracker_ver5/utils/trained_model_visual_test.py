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
    def __init__(self, root, dataset_dir, model_path, threshold=0.5):
        # Import tracker_model to register custom layers
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(script_dir)
        sys.path.append(os.path.dirname(script_dir))
        import tracker_model
        
        import tensorflow as tf
        
        self.threshold = threshold
        self.model_path = model_path
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
                if 256 in shape:
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
                "HeatmapNormalization": tracker_model.HeatmapNormalization,
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
        
        self.frames_frame = tk.Frame(self.root, bg="#121212")
        self.frames_frame.pack(pady=10)
        
        self.ref_panel, self.ref_lbl = self.create_frame_slot(self.frames_frame, "Reference Features")
        self.search_panel, self.search_lbl = self.create_frame_slot(self.frames_frame, "Search Frame")
        self.predicted_heatmap_panel, self.predicted_heatmap_lbl = self.create_frame_slot(self.frames_frame, "Predicted Heatmap")
        
        self.status_bar = tk.Label(self.root, text="Press [Space] for Next Sample", font=("Inter", 10), bg="#1c1c1c", fg="#aaaaaa", anchor="w", padx=15, pady=8)
        self.status_bar.pack(side="bottom", fill="x")
        
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
            
            # Prepare Target Label
            # Scale target_2d to normalized coordinates
            h_raw, w_raw = search_raw.shape[:2]
            if gt_quality > 0.5:
                # gt_coords is [y, x] in normalized [0, 1] space
                norm_x = gt_coords[1]
                norm_y = gt_coords[0]
                norm_coords = [norm_x, norm_y]
                curr_lbl_text = f"GT: [{norm_x:.2f}, {norm_y:.2f}]"
                curr_lbl_fg = "#00e6ff"
            else:
                norm_coords = None
                curr_lbl_text = "GT: None (Negative)"
                curr_lbl_fg = "#ff3366"
            
            # Show the largest reference crop (layer 0) scaled up
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
            
            # Prepare inputs for model
            if ref_stack.ndim == 4 and ref_stack.shape[0] == 1:
                ref_tensor = tf.cast(ref_stack, tf.float32)
            else:
                ref_tensor = tf.expand_dims(tf.cast(ref_stack, tf.float32), 0)
                
            if tf.reduce_max(ref_tensor) > 1.001:
                ref_tensor = ref_tensor / 255.0
            
            search_256 = cv2.resize(search_raw[:, :, 0], (256, 256), interpolation=cv2.INTER_LINEAR)
            if search_256.dtype == np.uint8:
                search_256_float = search_256.astype(np.float32) / 255.0
            else:
                search_256_float = search_256.astype(np.float32)
            search_tensor = tf.expand_dims(tf.expand_dims(search_256_float, -1), 0)
            
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
                    pred_heatmap = pred[0].numpy()[0]  # (256, 256, 1)
                    pred_quality = pred[1].numpy()[0][0]  # scalar float
                else:
                    pred_heatmap = pred.numpy()[0]
                    pred_quality = 1.0
            
            # Find peak
            heatmap = pred_heatmap[:, :, 0]
            flat_idx = np.argmax(heatmap)
            y_max, x_max = np.unravel_index(flat_idx, heatmap.shape)
            pred_norm = [x_max / heatmap.shape[1], y_max / heatmap.shape[0]]

 
            search_rgb = cv2.cvtColor(search_256, cv2.COLOR_GRAY2RGB)
            search_vis = (search_rgb.copy()*255).astype(np.uint8)
            
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
            
            # Prepare predicted heatmap color map
            heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            self.tk_img_predicted = ImageTk.PhotoImage(Image.fromarray(heatmap_color_rgb))
            
            self.ref_panel.config(image=self.tk_img_ref)
            self.search_panel.config(image=self.tk_img_search)
            self.predicted_heatmap_panel.config(image=self.tk_img_predicted)
            
            self.ref_lbl.config(text="Target Features")
            self.search_lbl.config(text=curr_lbl_text, fg=curr_lbl_fg)
            
            if norm_coords is not None:
                error = np.sqrt((pred_norm[0] - norm_coords[0])**2 + (pred_norm[1] - norm_coords[1])**2) * 256.0
                error_str = f"Error: {error:.1f}px"
            else:
                error_str = "Error: N/A"
            
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
    args = parser.parse_args()
    
    root = tk.Tk()
    app = ModelInferenceVisualizer(
        root, 
        args.dataset_dir, 
        args.model_path, 
        threshold=args.threshold
    )
    root.mainloop()

if __name__ == "__main__":
    main()
