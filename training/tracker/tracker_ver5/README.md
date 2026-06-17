# Tracker Ver 5

This sub-project introduces the **Tracker Ver 5** Siamese tracking architecture. Ver 5 transitions the network from pixel-level Gaussian heatmap regression to **direct coordinate regression** using a Soft-Argmax layer and Huber distance loss. It also introduces the grouped 2D cross-correlation attention mechanism (`conv2d_corr`).

### Key Highlights of Ver 5
* **Direct Coordinate Loss**: Eliminates 2D Gaussian heatmap target generation. The HDF5 dataset compiles targets directly as `ground_truth_coords` of shape `(None, 2)` containing `[y, x]` coordinates scaled to $256 \times 256$ space.
* **Soft-Argmax Regression**: The model outputs a spatial activation map which is passed through a differentiable Soft-Argmax layer to compute expected target coordinates. Huber loss is then computed directly on the Euclidean distance from the true target coordinate.
* **Grouped 2D Cross-Correlation Attention (`conv2d_corr`)**: A high-efficiency attention fusion layer that performs depthwise cross-correlation, sums across channels to produce a 1-channel similarity map, and projects/tiles it back to $C$ channels via a 1x1 2D convolution.
* **Unified Codebase**: Consolidates coordinates-based architectures into the main `tracker_model.py` and cleans up deprecated simulation code and unused FCN/pixel-shuffle models.

---

## Architectural Configuration (`model.conf`)

The network architecture is configured via the [model.conf](file:///home/elazarkin/work/projects/smart_rahfan/training/tracker/tracker_ver5/model.conf) file.

### Configuration Options
* **`reference_backbone` & `search_backbone`**:
  * `mini_mnv2` (Recommended): Lightweight MobileNetV2.
  * `mnv2_nano` (Recommended): Ultra-lightweight MobileNetV2.
  * `mnv1`: Standard MobileNetV1.
  * `mnv2`: Full MobileNetV2.
  * `yolo5`: CSPDarknet-style backbone.
  * `custom_legacy`: Original custom backbones.
* **`width_multiplier`**: Channel multiplier (default: `0.5`).
* **`attention_mechanism`**:
  * `conv2d_corr` (New in Ver 5): Grouped 2D cross-correlation (summed across channels) projected back via 1x1 convolution.
  * `depthwise_corr` (SiamFC-style): Depthwise cross-correlation with zero learnable parameters.
  * `dot_cross`: Single-head dot-product cross-attention.
  * `linear_cross`: Linearized cross-attention ($O(N)$ spatial complexity).
  * `multi_head_cross`: Multi-head cross-attention.
* **`decoder_type`**:
  * `fpn_add` (Recommended): FPN-style decoder with skip-add connections.
  * `unet`: U-Net decoder with skip-concatenations.
  * `pixel_shuffle`: Sub-pixel convolution decoder.
  * `light_naive`: Fast transposed convolutions without skip connections.

---

## Loss Formulation

### 1. Coordinate Distance Loss (Soft-Argmax + Huber)
The localization loss is computed directly on the expected coordinates predicted by the model's Soft-Argmax head:
$$\text{pred\_coords} = \text{Soft-Argmax}(\text{pred\_heatmap}, \beta=30.0)$$
$$\mathcal{L}_{\text{coords}} = \text{Huber}(\text{gt\_coords}, \text{pred\_coords})$$

For negative samples (missing target frames), the loss component is masked out using `ground_truth_quality`.

### 2. Localization Quality Score
A continuous confidence score from $0.0$ to $1.0$. The target quality score decays dynamically based on the distance error of the predicted target:
$$\text{Target Quality} = \max\left(1.0 - \frac{\text{Distance Error (px)}}{30.0}, 0.0\right)$$

---

## Running the Training Pipeline

### Step 1: Compile Dataset to Coordinates Format
```bash
python3 dataset_generator/dataset_compiler.py
```
This script compiles raw flights from the cache into `dataset_generator/compiled/dataset.h5`, storing `ground_truth_coords` `[y, x]` coordinates instead of 2D Gaussian heatmaps.

### Step 2: Run Stage 1 (Heatmap/Backbone Only)
Trains the spatial feature encoders and heatmap decoder:
```bash
python3 tracker_model.py train \
    --dataset_dir dataset_generator/compiled \
    --train_mode heatmap_only \
    --val_split 0.1 \
    --output outputs/tracker.keras \
    --batch_size 32
```

### Step 3: Run Stage 2 (Quality Only)
Freezes the pre-trained weights from Stage 1 and trains only the quality score head:
```bash
python3 tracker_model.py train \
    --dataset_dir dataset_generator/compiled \
    --train_mode quality_only \
    --init_keras_file outputs/tracker.keras \
    --val_split 0.1 \
    --output outputs/tracker.keras \
    --batch_size 32
```

---

## Live Inference Visualizer

Run the tkinter-based live visualizer to check predictions and ground truth labels:
```bash
python3 utils/trained_model_visual_test.py \
    --dataset_dir dataset_generator/compiled \
    --model_path outputs/tracker.keras
```
* **Target Features Panel**: Displays the current target template.
* **Search Frame Panel**: Displays the search frame with the Ground Truth coordinate (**cyan**) and the Model Prediction coordinate (**green**).
* **Predicted Heatmap Panel**: Displays the raw colorized heatmap output of the network.
