# TargetTrackerVer4 & TargetTrackerVerPixel Architecture Summary

This archive contains the architectural design, key parameters, and design patterns of the **Tracker Ver 4** model variants. This documentation serves as a reference for future restoration of specific features (e.g. Multi-Head Attention, Pixel Shuffle Decoder, and Shared Siamese Backbones).

---

## 1. TargetTrackerVer4 (SiamFC-style Dual Backbone)
Defined in `tracker_ver4/tracker_model.py`, this model separates feature extraction into two independent branches:
* **Reference Target Encoder**: A specialized CNN optimized for processing the $64 \times 64 \times 16$ multi-scale template stack.
* **Search Feature Extractor**: A CNN optimized for processing the $256 \times 256 \times 1$ search frame.

### Key Architectural Specs:
* **Siamese Split**: Different backbones can be selected for template and search branches (e.g., `mini_mnv2` for template, `mnv2_nano` for search).
* **Wide Configuration Space**: Configurable backbones (`mnv1`, `mnv2`, `mnv2_nano`, `yolo5`, `alex_net`), decoders (`fpn_add`, `unet`, `pixel_shuffle`, `light_naive`), and attention mechanisms (`depthwise_corr`, `dot_cross`, `linear_cross`, `multi_head_cross`).
* **Heatmap Normalization**: Ensures that predicted heatmaps are scaled relative to their peak activation, helping classification/regression stability.

---

## 2. TargetTrackerVerPixel (Shared Siamese Backbone & Pixel Shuffle)
Defined in `tracker_ver4_pxl.py`, this variant introduces a **single shared Siamese CNN backbone** for both template and search frames, and prioritizes **Pixel Shuffle (Sub-pixel Convolution)** for decoding.

### Key Architectural Specs:
* **Shared Feature Extractor**: Both the reference stack (resized to $128 \times 128$) and the search frame ($256 \times 256$) are passed through the exact same shared CNN. This forces the model to learn a unified representation space.
* **Reference Batch-Collapsing**: The reference stack is reshaped from `(B, 1, 64, 64, 16)` to a flat batch of size `B*16` before passing through the shared backbone. The output features are then reshaped back and average-pooled across the layers to form a single reference representation.
* **Pixel Shuffle Decoder**:
  * Leverages sub-pixel convolution (`DepthToSpace`) instead of bilinear upsampling or transpose convolutions.
  * Extrapolates channels into spatial coordinates to reconstruct high-resolution features without losing spatial detail.
  * Very fast and hardware-friendly on NPUs.

```python
# Pixel Shuffle Block Concept
def pixel_shuffle_block(inputs, out_filters, name_prefix):
    ps_c = out_filters * 4
    x_ps = layers.Conv2D(ps_c, (3, 3), padding="same", use_bias=False, name=f"{name_prefix}_ps_conv")(inputs)
    x_ps = _GroupNormalization(ps_c, name=f"{name_prefix}_ps_gn")(x_ps)
    x_ps = layers.ReLU(6.0, name=f"{name_prefix}_ps_relu")(x_ps)
    x_ps = DepthToSpace(block_size=2, name=f"{name_prefix}_ps_shuffle")(x_ps)
    return x_ps
```

---

## 3. Comparison Matrix

| Feature | TargetTrackerVer4 | TargetTrackerVerPixel |
| :--- | :--- | :--- |
| **Backbone Setup** | Dual Backbone (Ref Encoder + Search Extractor) | Single Shared Backbone |
| **Input Sizes** | Ref: $64 \times 64 \times 16$, Search: $256 \times 256 \times 1$ | Ref: $64 \times 64 \times 16$ (resized to $128 \times 128$), Search: $256 \times 256 \times 1$ |
| **Primary Attention** | Depthwise Cross-Correlation / Cross-Attention | Multi-Head Cross-Attention |
| **Decoder Type** | FPN Skip-Add / UNet | UNet / Pixel Shuffle (`DepthToSpace`) |
| **Branch Heads** | Heatmap + Quality Regression | Heatmap + Quality (Heatmap-Guided) |
| **Normalizations** | Capped Group Normalization / Batch Normalization | Capped Group Normalization / Batch Normalization |
