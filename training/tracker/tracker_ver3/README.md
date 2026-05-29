# Fully Convolutional Context-Preserving Tracker with Multi-Channel Masking (TargetTracker)

This directory contains the **Fully Convolutional Network (FCN) Siamese Tracker** (`tracker_model.py`).

By shifting from hard circular image cropping to a **multi-channel context-preserving input architecture**, this model retains the **entire peripheral background** across all frames. Rather than erasing pixels outside the target zone, the previous locations are fed to the model through a separate **attention mask channel**. 

This allows the network to learn rich **optical flow, translational motion, and camera ego-motion** cues from the background while maintaining precise focus on the target.

---

## 📐 Architecture Design

The tracker inputs three temporal branches. Each branch receives a 2-channel input (Channel 0 = Grayscale Image, Channel 1 = Attention Mask) of size $256 \times 256 \times 2$. A shared Siamese CNN backbone processes all three branches before temporal-fusion and spatial decoding.

```mermaid
graph TD
    %% Inputs
    subgraph Inputs ["Inputs (256x256x2 stacked)"]
        H_Branch["Hist Input<br>(Full frame + Hist soft mask)"]
        P_Branch["Prev Input<br>(Full frame + Prev soft mask)"]
        C_Branch["Curr Input<br>(Full frame + All-zeros mask)"]
    end

    %% Siamese Feature Extraction
    subgraph Siamese_Backbone ["Siamese CNN Feature Extractor (Shared Weights)"]
        CNN_H["Siamese CNN<br>(Accepts 2 channels)"]
        CNN_P["Siamese CNN<br>(Accepts 2 channels)"]
        CNN_C["Siamese CNN<br>(Accepts 2 channels)"]
    end

    H_Branch --> CNN_H
    P_Branch --> CNN_P
    C_Branch --> CNN_C

    %% CNN Feature Maps
    CNN_H --> F_H["Hist Features<br>(16, 16, 256)"]
    CNN_P --> F_P["Prev Features<br>(16, 16, 256)"]
    CNN_C --> F_C["Curr Features<br>(16, 16, 256)"]

    %% Temporal and Motion Fusion
    subgraph Motion_Dynamics ["Motion Dynamics & Temporal Fusion"]
        Sub_H2P["Subtract (Prev - Hist)"]
        Sub_P2C["Subtract (Curr - Prev)"]
        
        F_H --> Sub_H2P
        F_P --> Sub_H2P
        
        F_P --> Sub_P2C
        F_C --> Sub_P2C
        
        Concat_Temp["Concatenate Channel-wise"]
        
        F_H --> Concat_Temp
        F_P --> Concat_Temp
        F_C --> Concat_Temp
        Sub_H2P --> Concat_Temp
        Sub_P2C --> Concat_Temp
    end

    %% FCN Decoder
    subgraph Decoder ["FCN Decoder & Upsampler"]
        Disp_Conv["Conv 3x3 + BN + ReLU<br>(256 channels)"]
        Upsample_1["Conv2DTranspose (128)<br>Strides = 2 (32x32)"]
        Upsample_2["Conv2DTranspose (64)<br>Strides = 2 (64x64)"]
        Final_Conv["Conv 3x3 + Sigmoid (1)<br>(64x64x1)"]
        
        Concat_Temp --> Disp_Conv
        Disp_Conv --> Upsample_1
        Upsample_1 --> Upsample_2
        Upsample_2 --> Final_Conv
    end

    %% Outputs
    Final_Conv --> Heatmap["Predicted Heatmap<br>(64, 64, 1)"]
```

---

## 🛠️ Key Architectural Paradigms

### 1. Multi-Channel Siamese Feature Extraction
Unlike traditional trackers that feed single-channel images, `TargetTracker`'s Siamese CNN takes **2-channel inputs**:
* **Channel 0**: The full, unmasked grayscale image. This preserves all context, allowing low-level and high-level filters to track global background movement.
* **Channel 1**: An attention mask pinpointing the target location in historical and previous frames. For the search frame (`curr`), this mask channel is filled with **zeros** since the target position is unknown.

### 2. Flexible Masking Strategies
To prevent artificial high-frequency edges (which binary circular masks suffer from and can confuse CNN filters), the model provides helpers for two types of attention masking:
* **Circular Mask (`generate_circular_mask`)**: Hard-edge binary circle (1.0 inside, 0.0 outside) representing a local search region.
* **Gaussian Soft Mask (`generate_gaussian_mask`)**: Smooth Gaussian heatmap with tunable standard deviation ($\sigma$). This smoothly guides the model's focus to the region of interest without creating harsh artificial borders.

### 3. Separation of Concerns (Dataset Generation)
Dataset generation logic is fully decoupled from the model script, leaving `tracker_model.py` entirely focused on model definition, custom loss functions, and high-performance training loops.

---

## 🏋️ Custom Spatial Loss Functions
The model fully registers and supports custom loss functions tailored for spatial heatmap regression:
1. **`dice_bce`**: Combines Dice Loss (structural overlap) with Binary Cross Entropy (pixel-wise convergence). Excellent for handling class imbalance.
2. **`focal`**: Sigmoid Focal Loss, designed specifically to focus gradients on hard, active target pixels while suppressing easy background zeros.

---

## 🚀 Execution & Training Guide

### 🏋️ Train the FCN Model
To train the model, ensure your external dataset generator outputs `.pkl` batches containing:
- **`inputs`**: List of 3 numpy arrays, each of shape `(batch_size, 256, 256, 2)` (stacked image and mask).
- **`targets`**: A numpy array of shape `(batch_size, 64, 64, 1)` (Gaussian target heatmap).

Run training using:
```bash
python3 training/tracker/tracker_ver3/tracker_model.py train \
    --dataset_dir /path/to/2channel_dataset \
    --lr 0.001 \
    --num_of_epochs 100 \
    --loss dice_bce \
    --output model_tracker.keras
```
