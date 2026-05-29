#!/bin/env python3
import os
import sys
import gc
import random
import pickle
import argparse
import numpy as np
import cv2
import json
import time
import hashlib


# =====================================================================
# Heatmap and Attention Masking Helpers (Matching tracker_model exactly)
# =====================================================================

def generate_attention_mask(coords, size=256, mask_type='gaussian', radius=128, sigma=15.0):
    """
    Generates a 2D attention mask centered around the normalized coordinate.
    
    Args:
        coords (list/tuple): Normalized [x, y] coordinates in [0, 1] range.
        size (int): Dimensions of the output mask grid (default: 256).
        mask_type (str): 'circular' or 'gaussian' (default: 'gaussian').
        radius (int): Radius of the circular mask in pixels (only used in 'circular' mode).
        sigma (float): Standard deviation of the Gaussian mask in pixels (only used in 'gaussian' mode).
        
    Returns:
        np.ndarray: Attention mask array of shape (size, size, 1) in [0.0, 1.0].
    """
    h, w = size, size
    x_target = coords[0] * w
    y_target = coords[1] * h
    
    if mask_type == 'circular':
        mask = np.zeros((h, w, 1), dtype=np.float32)
        cv2.circle(mask, (int(x_target), int(y_target)), int(radius), 1.0, -1)
        return mask
    else:
        # Gaussian soft mask
        x = np.arange(0, w, 1, dtype=np.float32)
        y = np.arange(0, h, 1, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x, y)
        d2 = (x_grid - x_target) ** 2 + (y_grid - y_target) ** 2
        mask = np.exp(-d2 / (2.0 * (sigma ** 2)))
        return np.expand_dims(mask, axis=-1)

def generate_gaussian_heatmap(coords, size=64, sigma=4.0):
    """
    Generates a 2D Gaussian heatmap centered around the normalized coordinate.
    
    Args:
        coords (list/tuple): Normalized [x, y] coordinate in [0, 1] range.
        size (int): Dimensions of the output heatmap grid (default: 64).
        sigma (float): Standard deviation of the Gaussian kernel (default: 4.0).
        
    Returns:
        np.ndarray: Gaussian heatmap array of shape (size, size, 1) in [0.0, 1.0].
    """
    x_target = coords[0] * size
    y_target = coords[1] * size
    
    x = np.arange(0, size, 1, dtype=np.float32)
    y = np.arange(0, size, 1, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)
    
    d2 = (x_grid - x_target) ** 2 + (y_grid - y_target) ** 2
    heatmap = np.exp(-d2 / (2.0 * sigma ** 2))
    return np.expand_dims(heatmap, axis=-1)

# =====================================================================
# Affine SIFT (ASIFT) Perspective Simulation Helpers
# =====================================================================

def get_affine_tilt_matrix(w, h, t, phi):
    """
    Constructs the 2x3 affine matrix for a viewpoint tilt t and rotation phi,
    along with its inverse for keypoint back-projection.
    """
    center = (w / 2.0, h / 2.0)
    
    # 1. Rotation matrix around center
    R = cv2.getRotationMatrix2D(center, phi, 1.0)
    
    # 2. Horizontal scaling matrix (simulating camera tilt along x-axis)
    S = np.array([
        [1.0 / t, 0.0, center[0] * (1.0 - 1.0 / t)],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)
    
    # 3. Combine matrices using homogenous 3x3 coordinates
    R_3x3 = np.vstack([R, [0.0, 0.0, 1.0]])
    S_3x3 = np.vstack([S, [0.0, 0.0, 1.0]])
    
    M_3x3 = np.dot(S_3x3, R_3x3)
    M = M_3x3[:2, :]
    
    # Inverse matrix for projecting coordinates back to the original frame
    M_inv_3x3 = np.linalg.inv(M_3x3)
    M_inv = M_inv_3x3[:2, :]
    
    return M, M_inv

class ASIFTMatcher:
    """
    Highly optimized and accelerated Affine SIFT implementation using OpenCV.
    Simulates affine camera viewpoints, aggregates keypoints/descriptors,
    and performs a single global match.
    """
    def __init__(self, feature_type='sift'):
        self.feature_type = feature_type
        if feature_type == 'surf':
            self.detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        else:
            self.detector = cv2.SIFT_create()
            
        # 5 highly representative viewpoints for forward progress & gimbal tilts
        self.views = [
            (1.0, 0.0),       # Standard view (no warp)
            (1.414, 0.0),     # 45 deg tilt along X-axis
            (1.414, 90.0),    # 45 deg tilt along Y-axis
            (2.0, 0.0),       # 60 deg tilt along X-axis
            (2.0, 90.0)       # 60 deg tilt along Y-axis
        ]

    def detect_and_backproject(self, img):
        """
        Warps the image for all simulated viewpoints, extracts keypoints/descriptors,
        and projects keypoint coordinates back to the original image space.
        """
        h, w = img.shape[:2]
        aggregated_kps = []
        aggregated_des = []
        
        for t, phi in self.views:
            if t == 1.0 and phi == 0.0:
                kps, des = self.detector.detectAndCompute(img, None)
                if des is not None:
                    for kp in kps:
                        aggregated_kps.append(kp)
                    aggregated_des.append(des)
            else:
                M, M_inv = get_affine_tilt_matrix(w, h, t, phi)
                
                # Perform GPU-equivalent multithreaded CPU warp
                img_warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                
                kps, des = self.detector.detectAndCompute(img_warped, None)
                if des is not None:
                    for kp in kps:
                        pt_w = np.array([kp.pt[0], kp.pt[1], 1.0], dtype=np.float32)
                        pt_orig = np.dot(M_inv, pt_w)
                        
                        # Instantiation using positional constructor for maximum version compatibility
                        kp_orig = cv2.KeyPoint(
                            float(pt_orig[0]),
                            float(pt_orig[1]),
                            kp.size,
                            kp.angle + phi,
                            kp.response,
                            kp.octave,
                            kp.class_id
                        )
                        aggregated_kps.append(kp_orig)
                    aggregated_des.append(des)
                    
        if len(aggregated_des) == 0:
            return [], None
            
        return aggregated_kps, np.vstack(aggregated_des)

# =====================================================================
# Video Crawling and Geometric Invariance Helpers
# =====================================================================

def to_grayscale(frame_raw):
    """
    Safely converts a raw video frame to grayscale, handling both BGR
    and pre-existing grayscale single-channel layouts.
    """
    if frame_raw is None:
        return None
    if len(frame_raw.shape) == 2:
        return frame_raw.copy()
    return cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)

def find_video_files(directory):
    """
    Recursively crawls the directory using os.walk to find all video files.
    """
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".mpeg", ".mpg", ".webm")
    video_paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(video_extensions):
                video_paths.append(os.path.join(root, f))
    return video_paths

def extract_features(img, feature_type='asift', asift_matcher=None):
    """
    Extracts keypoints and descriptors based on the selected feature type.
    """
    if feature_type == 'asift':
        if asift_matcher is None:
            asift_matcher = ASIFTMatcher(feature_type='sift')
        return asift_matcher.detect_and_backproject(img)
    elif feature_type == 'surf':
        detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        kp, des = detector.detectAndCompute(img, None)
        return kp, des
    else:  # sift
        detector = cv2.SIFT_create()
        kp, des = detector.detectAndCompute(img, None)
        return kp, des

def compute_epipolar_distance(pt1, pt2, F):
    """
    Computes the orthogonal distance from pt2 (in image 2) to the epipolar line
    corresponding to pt1 (in image 1) given by l = F * [pt1[0], pt1[1], 1.0]^T.
    """
    h_pt1 = np.array([pt1[0], pt1[1], 1.0], dtype=np.float32)
    line = np.dot(F, h_pt1)
    a, b, c = line[0], line[1], line[2]
    denom = np.sqrt(a**2 + b**2)
    if denom < 1e-8:
        return 999.0
    return abs(a * pt2[0] + b * pt2[1] + c) / denom

def match_features_triplet(f_hist, f_prev, f_curr, ratio=0.85, min_inliers=6, feature_type='asift', asift_matcher=None,
                           ransac_thresh=5.0, min_motion_pc=1.0, min_motion_hp=3.0, min_texture_std=3.0, proc_size=800, min_ncc=0.75,
                           keep_debug_info=False):
    """
    Runs SIFT/ASIFT/SURF keypoint detection and matches features across three frames
    (hist -> prev and prev -> curr). Fits a Fundamental Matrix via RANSAC
    to reject dynamic outliers and returns a list of verified inlier coordinate paths.
    
    If inliers < min_inliers, it still returns the detected raw matches and status="failed"
    to allow the visualization debug panel to render the failure diagnostic.
    """
    kp_hist, des_hist = extract_features(f_hist, feature_type, asift_matcher)
    kp_prev, des_prev = extract_features(f_prev, feature_type, asift_matcher)
    kp_curr, des_curr = extract_features(f_curr, feature_type, asift_matcher)
    
    if des_hist is None or des_prev is None or des_curr is None:
        return {
            "status": "failed",
            "reason": "empty_descriptors",
            "paths": [],
            "kp_hist": kp_hist if (kp_hist and keep_debug_info) else [],
            "kp_prev": kp_prev if (kp_prev and keep_debug_info) else [],
            "kp_curr": kp_curr if (kp_curr and keep_debug_info) else [],
            "triplets": [],
            "inliers": []
        }
        
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    # Match hist -> prev
    matches_12 = bf.knnMatch(des_hist, des_prev, k=2)
    hist_to_prev = {}
    for m_list in matches_12:
        if len(m_list) == 2:
            m, n = m_list
            if m.distance < ratio * n.distance:
                hist_to_prev[m.queryIdx] = m.trainIdx
                
    # Match prev -> curr
    matches_23 = bf.knnMatch(des_prev, des_curr, k=2)
    prev_to_curr = {}
    for m_list in matches_23:
        if len(m_list) == 2:
            m, n = m_list
            if m.distance < ratio * n.distance:
                prev_to_curr[m.queryIdx] = m.trainIdx
                
    # Scale spatial motion/texture parameters to proc_size space dynamically
    scale_factor = proc_size / 256.0
    scaled_min_motion_pc = min_motion_pc * scale_factor
    scaled_min_motion_hp = min_motion_hp * scale_factor
    
    # Scale patch size for texture filtering (default 5px in 256x256 becomes 15px in 800x800)
    r = int(round(5 * scale_factor))
    
    # Find matching keypoint triplets (hist -> prev -> curr)
    pts_hist, pts_prev, pts_curr = [], [], []
    kp_triplets = []
    
    for idx_hist, idx_prev in hist_to_prev.items():
        if idx_prev in prev_to_curr:
            idx_curr = prev_to_curr[idx_prev]
            pt_h = kp_hist[idx_hist].pt
            pt_p = kp_prev[idx_prev].pt
            pt_c = kp_curr[idx_curr].pt
            
            # 1. Filter A: Motion Magnitude (Velocity)
            v_pc = np.array([pt_c[0] - pt_p[0], pt_c[1] - pt_p[1]], dtype=np.float32)
            v_hp = np.array([pt_p[0] - pt_h[0], pt_p[1] - pt_h[1]], dtype=np.float32)
            
            dist_pc = np.sqrt(v_pc[0]**2 + v_pc[1]**2)
            dist_hp = np.sqrt(v_hp[0]**2 + v_hp[1]**2)
            
            # Eased static filters (using dynamically scaled thresholds)
            if dist_pc < scaled_min_motion_pc or dist_hp < scaled_min_motion_hp:
                continue
                
            # 2. Filter B: Directional Coherence (Cosine Similarity)
            dot_product = v_hp[0] * v_pc[0] + v_hp[1] * v_pc[1]
            cos_theta = dot_product / (dist_hp * dist_pc + 1e-8)
            
            # Cosine similarity must be > 0.0 to filter out oscillations/vibrations
            if cos_theta < 0.0:
                continue
                
            # 3. Filter C: Local Texture/Variance Filter
            px, py = int(pt_p[0]), int(pt_p[1])
            if py - r >= 0 and py + r + 1 <= proc_size and px - r >= 0 and px + r + 1 <= proc_size:
                patch = f_prev[py-r : py+r+1, px-r : px+r+1]
                patch_std = np.std(patch)
                # Eased texture filter
                if patch_std < min_texture_std:
                    continue
            else:
                continue
                
            # 4. Filter D: Patch Normalized Cross-Correlation (NCC) to reject false SIFT matches
            r_ncc = 10  # 21x21 patch size
            hx, hy = int(pt_h[0]), int(pt_h[1])
            cx, cy = int(pt_c[0]), int(pt_c[1])
            
            if (hy - r_ncc >= 0 and hy + r_ncc + 1 <= proc_size and hx - r_ncc >= 0 and hx + r_ncc + 1 <= proc_size and
                py - r_ncc >= 0 and py + r_ncc + 1 <= proc_size and px - r_ncc >= 0 and px + r_ncc + 1 <= proc_size and
                cy - r_ncc >= 0 and cy + r_ncc + 1 <= proc_size and cx - r_ncc >= 0 and cx + r_ncc + 1 <= proc_size):
                
                patch_h = f_hist[hy-r_ncc : hy+r_ncc+1, hx-r_ncc : hx+r_ncc+1]
                patch_p = f_prev[py-r_ncc : py+r_ncc+1, px-r_ncc : px+r_ncc+1]
                patch_c = f_curr[cy-r_ncc : cy+r_ncc+1, cx-r_ncc : cx+r_ncc+1]
                
                if np.std(patch_h) >= 1e-3 and np.std(patch_p) >= 1e-3 and np.std(patch_c) >= 1e-3:
                    res_hp = cv2.matchTemplate(patch_h, patch_p, cv2.TM_CCOEFF_NORMED)
                    ncc_hp = res_hp[0, 0]
                    
                    res_pc = cv2.matchTemplate(patch_p, patch_c, cv2.TM_CCOEFF_NORMED)
                    ncc_pc = res_pc[0, 0]
                    
                    if ncc_hp < min_ncc or ncc_pc < min_ncc:
                        continue
                else:
                    continue
            else:
                continue
                
            pts_hist.append(pt_h)
            pts_prev.append(pt_p)
            pts_curr.append(pt_c)
            kp_triplets.append((idx_hist, idx_prev, idx_curr))
            
    # Clear SIFT keypoints if not needed to save massive memory
    if not keep_debug_info:
        kp_hist = []
        kp_prev = []
        kp_curr = []

    if len(pts_hist) < min_inliers:
        return {
            "status": "failed",
            "reason": "too_few_raw_triplets",
            "paths": [],
            "kp_hist": kp_hist,
            "kp_prev": kp_prev,
            "kp_curr": kp_curr,
            "triplets": kp_triplets,
            "inliers": list(range(len(pts_hist)))  # return all raw matching indexes
        }
        
    pts_hist = np.float32(pts_hist)
    pts_prev = np.float32(pts_prev)
    pts_curr = np.float32(pts_curr)
    
    # Fit Fundamental Matrices using RANSAC to verify epipolar constraints
    F_12, mask_12 = cv2.findFundamentalMat(pts_hist, pts_prev, cv2.FM_RANSAC, ransac_thresh)
    F_23, mask_23 = cv2.findFundamentalMat(pts_prev, pts_curr, cv2.FM_RANSAC, ransac_thresh)
    
    if F_12 is None or F_23 is None or mask_12 is None or mask_23 is None or F_12.shape != (3, 3) or F_23.shape != (3, 3):
        return {
            "status": "failed",
            "reason": "ransac_matrix_failure",
            "paths": [],
            "kp_hist": kp_hist,
            "kp_prev": kp_prev,
            "kp_curr": kp_curr,
            "triplets": kp_triplets,
            "inliers": list(range(len(pts_hist)))
        }
        
    m12_flat = mask_12.ravel()
    m23_flat = mask_23.ravel()
    
    inlier_indices = []
    for i in range(len(pts_hist)):
        if m12_flat[i] == 1 and m23_flat[i] == 1:
            inlier_indices.append(i)
            
    if len(inlier_indices) < min_inliers:
        return {
            "status": "failed",
            "reason": "too_few_ransac_inliers",
            "paths": [],
            "kp_hist": kp_hist,
            "kp_prev": kp_prev,
            "kp_curr": kp_curr,
            "triplets": kp_triplets,
            "inliers": inlier_indices
        }
        
    # Calculate epipolar distance errors for each inlier and sort them
    inliers_with_errors = []
    for idx in inlier_indices:
        pt_h = pts_hist[idx]
        pt_p = pts_prev[idx]
        pt_c = pts_curr[idx]
        
        d12 = compute_epipolar_distance(pt_h, pt_p, F_12)
        d23 = compute_epipolar_distance(pt_p, pt_c, F_23)
        total_err = d12 + d23
        inliers_with_errors.append((idx, total_err))
        
    # Sort inliers by total epipolar error in ascending order (best first)
    inliers_with_errors.sort(key=lambda item: item[1])
    sorted_inlier_indices = [item[0] for item in inliers_with_errors]
    
    # Construct verified normalized paths in sorted order (best first)
    verified_paths = []
    for idx in sorted_inlier_indices:
        h_pt = pts_hist[idx]
        p_pt = pts_prev[idx]
        c_pt = pts_curr[idx]
        
        path = {
            "hist": [np.clip(h_pt[0] / proc_size, 0.0, 1.0), np.clip(h_pt[1] / proc_size, 0.0, 1.0)],
            "prev": [np.clip(p_pt[0] / proc_size, 0.0, 1.0), np.clip(p_pt[1] / proc_size, 0.0, 1.0)],
            "curr": [np.clip(c_pt[0] / proc_size, 0.0, 1.0), np.clip(c_pt[1] / proc_size, 0.0, 1.0)]
        }
        verified_paths.append(path)
        
    return {
        "status": "success",
        "paths": verified_paths,
        "kp_hist": kp_hist,
        "kp_prev": kp_prev,
        "kp_curr": kp_curr,
        "triplets": kp_triplets,
        "inliers": sorted_inlier_indices
    }

def simulate_hover_jitter(frame_256, coords):
    """
    Simulates drone hovering and local camera jitter (gimbal vibrations, slight wind drift)
    by applying small random rotation and translation matrices to a static frame.
    
    Args:
        frame_256 (np.ndarray): Grayscale frame resized to 256x256.
        coords (list/tuple): Normalized target coordinate [x, y].
        
    Returns:
        tuple: (warped_frame, warped_coords_normalized)
    """
    theta = np.random.uniform(-3.0, 3.0)  # Random rotation within +/- 3 degrees
    dx = np.random.uniform(-5.0, 5.0)    # Random translation within +/- 5 pixels
    dy = np.random.uniform(-5.0, 5.0)
    
    M = cv2.getRotationMatrix2D((128.0, 128.0), theta, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    
    warped_frame = cv2.warpAffine(frame_256, M, (256, 256), borderMode=cv2.BORDER_REPLICATE)
    
    # Project coordinates
    x_px = coords[0] * 256.0
    y_px = coords[1] * 256.0
    pt = np.array([x_px, y_px, 1.0], dtype=np.float32)
    warped_pt = np.dot(M, pt)
    
    warped_norm = [
        np.clip(warped_pt[0] / 256.0, 0.0, 1.0),
        np.clip(warped_pt[1] / 256.0, 0.0, 1.0)
    ]
    return warped_frame, warped_norm

def is_path_in_inner_2_3(path):
    """
    Checks if a given path is entirely located within the inner 2/3 region
    of the image space [0.167, 0.833] across all three frames.
    This helps avoid lens distortion, which increases towards the frame boundaries.
    """
    for frame in ["hist", "prev", "curr"]:
        x, y = path[frame]
        if not (0.167 <= x <= 0.833 and 0.167 <= y <= 0.833):
            return False
    return True

# =====================================================================
# Dashboard Rendering and Visualization
# =====================================================================

def draw_hud_label(img, label, org, color=(0, 255, 0)):
    """
    Draws a highly aesthetic HUD overlay label with a semi-transparent background block.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    (w, h), baseline = cv2.getTextSize(label, font, scale, thickness)
    
    y1 = max(0, org[1] - h - 4)
    y2 = min(img.shape[0], org[1] + baseline)
    x1 = max(0, org[0] - 4)
    x2 = min(img.shape[1], org[0] + w + 4)
    
    sub_img = img[y1:y2, x1:x2]
    if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
        rect = np.zeros_like(sub_img)
        rect[:] = 15  # Very dark grey
        cv2.addWeighted(sub_img, 0.3, rect, 0.7, 0, sub_img)
    
    cv2.putText(img, label, org, font, scale, color, thickness, cv2.LINE_AA)

def draw_epipolar_line(img, line, color, thickness=1):
    """
    Highly robust utility to draw an epipolar line ax + by + c = 0 on an image.
    Avoids division by zero and correctly handles nearly vertical/horizontal lines.
    """
    a, b, c = line[0], line[1], line[2]
    h, w = img.shape[:2]
    if abs(b) > abs(a):
        # Line is more horizontal, solve for y at x=0 and x=w
        x0 = 0
        y0 = int(round(-c / b))
        x1 = w
        y1 = int(round(-(c + a * w) / b))
    else:
        # Line is more vertical, solve for x at y=0 and y=h
        y0 = 0
        x0 = int(round(-c / a))
        y1 = h
        x1 = int(round(-(c + b * h) / a))
    cv2.line(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)

def render_dashboard(f_hist_256, f_prev_256, f_curr_256, hist_mask, prev_mask, hist_norm, prev_norm, curr_norm, sift_info=None, proc_size=800, feature_type='asift', label_radius=32):
    """
    Renders a stunning 4x3 widescreen grid dashboard containing full frames overlaid with highly transparent
    colored attention masks (50% opacity), green target indicators, and inter-image keypoint matching lines.
    Additionally displays two real-time epipolar diagnostic panels, a telemetry console,
    and a pure attention masks & expected heatmap row (Row 4).
    """
    # 1. Convert grayscale images (Channel 0) to BGR for colorful HUD overlays
    h_color = cv2.cvtColor(f_hist_256, cv2.COLOR_GRAY2BGR)
    p_color = cv2.cvtColor(f_prev_256, cv2.COLOR_GRAY2BGR)
    c_color = cv2.cvtColor(f_curr_256, cv2.COLOR_GRAY2BGR)
    
    # 2. Blend the attention masks (Channel 1) onto Hist and Prev frames
    # hist_mask is (256, 256, 1) float32 in [0.0, 1.0]. Convert to red overlay (0, 0, 255)
    h_mask_bgr = np.zeros_like(h_color)
    h_mask_bgr[:, :, 2] = (hist_mask[:, :, 0] * 255.0).astype(np.uint8)  # Red channel
    
    # prev_mask is (256, 256, 1) float32 in [0.0, 1.0]. Convert to blue overlay (255, 0, 0)
    p_mask_bgr = np.zeros_like(p_color)
    p_mask_bgr[:, :, 0] = (prev_mask[:, :, 0] * 255.0).astype(np.uint8)  # Blue channel
    
    # 50% opacity blend as requested by the user
    cv2.addWeighted(h_mask_bgr, 0.50, h_color, 1.0, 0, h_color)
    cv2.addWeighted(p_mask_bgr, 0.50, p_color, 1.0, 0, p_color)
    
    # 3. Draw target indicators
    hx, hy = int(hist_norm[0] * 256), int(hist_norm[1] * 256)
    cv2.circle(h_color, (hx, hy), 4, (0, 0, 255), -1)  # Red center dot
    
    px, py = int(prev_norm[0] * 256), int(prev_norm[1] * 256)
    cv2.circle(p_color, (px, py), 4, (255, 0, 0), -1)  # Blue center dot
    
    cx, cy = int(curr_norm[0] * 256), int(curr_norm[1] * 256)
    cv2.circle(c_color, (cx, cy), 8, (0, 255, 0), 2)   # Neon green target ring
    cv2.circle(c_color, (cx, cy), 2, (0, 255, 0), -1)  # Center dot
    
    # 4. SIFT/ASIFT/SURF match visualization panel (across hist -> prev -> curr)
    is_success = (sift_info is not None and sift_info.get("status") == "success")
    
    # Prepare the triple-wide horizontal panel (hist + prev + curr)
    sift_panel = np.hstack([f_hist_256, f_prev_256, f_curr_256])
    s_color = cv2.cvtColor(sift_panel, cv2.COLOR_GRAY2BGR)
    
    # Draw connections in all visualization states (success and failure)
    if sift_info is not None and len(sift_info.get("inliers", [])) > 0:
        kp_hist = sift_info["kp_hist"]
        kp_prev = sift_info["kp_prev"]
        kp_curr = sift_info["kp_curr"]
        triplets = sift_info["triplets"]
        inliers = sift_info["inliers"]
        
        # Scale coordinates from proc_size back to 256x256 visualization space
        scale = 256.0 / proc_size
        
        # Draw other matches (color_idx > 0) first (background)
        for color_idx, idx in enumerate(inliers):
            if color_idx > 0 and idx < len(triplets):
                trip = triplets[idx]
                pt_h = kp_hist[trip[0]].pt
                pt_p = kp_prev[trip[1]].pt
                pt_c = kp_curr[trip[2]].pt
                
                p1_h = (int(pt_h[0] * scale), int(pt_h[1] * scale))
                p2_p = (int(pt_p[0] * scale + 256), int(pt_p[1] * scale))  # Shifted right by 256 (prev frame)
                p3_c = (int(pt_c[0] * scale + 512), int(pt_c[1] * scale))  # Shifted right by 512 (curr frame)
                
                color = (255, 255, 255)
                thickness = 1
                radius = 3
                
                cv2.line(s_color, p1_h, p2_p, color, thickness, cv2.LINE_AA)
                cv2.line(s_color, p2_p, p3_c, color, thickness, cv2.LINE_AA)
                cv2.circle(s_color, p1_h, radius, color, -1)
                cv2.circle(s_color, p2_p, radius, color, -1)
                cv2.circle(s_color, p3_c, radius, color, -1)
                
        # Draw selected optimal match (color_idx == 0) last (foreground)
        if len(inliers) > 0:
            idx = inliers[0]
            if idx < len(triplets):
                trip = triplets[idx]
                pt_h = kp_hist[trip[0]].pt
                pt_p = kp_prev[trip[1]].pt
                pt_c = kp_curr[trip[2]].pt
                
                p1_h = (int(pt_h[0] * scale), int(pt_h[1] * scale))
                p2_p = (int(pt_p[0] * scale + 256), int(pt_p[1] * scale))
                p3_c = (int(pt_c[0] * scale + 512), int(pt_c[1] * scale))
                
                color = (255, 0, 0)
                thickness = 2
                radius = 5
                
                cv2.line(s_color, p1_h, p2_p, color, thickness, cv2.LINE_AA)
                cv2.line(s_color, p2_p, p3_c, color, thickness, cv2.LINE_AA)
                cv2.circle(s_color, p1_h, radius, color, -1)
                cv2.circle(s_color, p2_p, radius, color, -1)
                cv2.circle(s_color, p3_c, radius, color, -1)
    else:
        # Hover Mode or Empty matching (centered on 768px widescreen canvas)
        if sift_info is None:
            cv2.putText(s_color, "HOVER MODE", (280, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(s_color, "Gimbal Jitter Active", (300, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(s_color, "NO MATCHES FOUND", (260, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
    # 5. Row 3: Epipolar lines and Telemetry
    epipoles_on_prev = cv2.cvtColor(f_prev_256, cv2.COLOR_GRAY2BGR)
    epipoles_on_hist = cv2.cvtColor(f_hist_256, cv2.COLOR_GRAY2BGR)
    telemetry_panel = np.zeros((256, 256, 3), dtype=np.uint8) + 20  # dark charcoal (20, 20, 20)
    
    avg_err = 0.0
    inlier_count = 0
    
    if sift_info is not None:
        inliers = sift_info.get("inliers", [])
        triplets = sift_info.get("triplets", [])
        kp_hist = sift_info.get("kp_hist", [])
        kp_prev = sift_info.get("kp_prev", [])
        
        if len(inliers) > 0 and len(triplets) > 0:
            scale = 256.0 / proc_size
            pts_h_256 = []
            pts_p_256 = []
            
            for idx in inliers:
                if idx < len(triplets):
                    trip = triplets[idx]
                    pt_h = kp_hist[trip[0]].pt
                    pt_p = kp_prev[trip[1]].pt
                    pts_h_256.append([pt_h[0] * scale, pt_h[1] * scale])
                    pts_p_256.append([pt_p[0] * scale, pt_p[1] * scale])
                    
            if len(pts_h_256) >= 5:
                pts_h_256 = np.float32(pts_h_256)
                pts_p_256 = np.float32(pts_p_256)
                
                # Fit fundamental matrix in 256x256 space
                if len(pts_h_256) >= 8:
                    F_256, _ = cv2.findFundamentalMat(pts_h_256, pts_p_256, cv2.FM_8POINT)
                else:
                    F_256, _ = cv2.findFundamentalMat(pts_h_256, pts_p_256, cv2.FM_LMEDS)
                    
                if F_256 is not None and F_256.shape == (3, 3):
                    # Compute epipolar lines in both images
                    lines_on_prev = cv2.computeCorrespondEpilines(pts_h_256.reshape(-1, 1, 2), 1, F_256)
                    lines_on_hist = cv2.computeCorrespondEpilines(pts_p_256.reshape(-1, 1, 2), 2, F_256)
                    
                    total_err = 0.0
                    err_count = 0
                    
                    # Draw other matches (i > 0) first (background)
                    for i in range(1, len(pts_h_256)):
                        pt_h = pts_h_256[i]
                        pt_p = pts_p_256[i]
                        
                        l_prev = lines_on_prev[i][0]
                        l_hist = lines_on_hist[i][0]
                        
                        # Calculate orthogonal distance for error metric
                        a, b, c = l_prev[0], l_prev[1], l_prev[2]
                        denom = np.sqrt(a**2 + b**2)
                        if denom > 1e-8:
                            dist = abs(a * pt_p[0] + b * pt_p[1] + c) / denom
                            total_err += dist
                            err_count += 1
                            
                        color = (255, 255, 255)
                        thickness = 1
                        radius = 3
                        
                        draw_epipolar_line(epipoles_on_prev, l_prev, color, thickness)
                        cv2.circle(epipoles_on_prev, (int(pt_p[0]), int(pt_p[1])), radius, color, -1)
                        
                        draw_epipolar_line(epipoles_on_hist, l_hist, color, thickness)
                        cv2.circle(epipoles_on_hist, (int(pt_h[0]), int(pt_h[1])), radius, color, -1)
                        
                    # Draw selected optimal match (i == 0) last (foreground)
                    if len(pts_h_256) > 0:
                        pt_h = pts_h_256[0]
                        pt_p = pts_p_256[0]
                        
                        l_prev = lines_on_prev[0][0]
                        l_hist = lines_on_hist[0][0]
                        
                        # Calculate orthogonal distance for error metric
                        a, b, c = l_prev[0], l_prev[1], l_prev[2]
                        denom = np.sqrt(a**2 + b**2)
                        if denom > 1e-8:
                            dist = abs(a * pt_p[0] + b * pt_p[1] + c) / denom
                            total_err += dist
                            err_count += 1
                            
                        color = (255, 0, 0)
                        thickness = 2
                        radius = 5
                        
                        draw_epipolar_line(epipoles_on_prev, l_prev, color, thickness)
                        cv2.circle(epipoles_on_prev, (int(pt_p[0]), int(pt_p[1])), radius, color, -1)
                        
                        draw_epipolar_line(epipoles_on_hist, l_hist, color, thickness)
                        cv2.circle(epipoles_on_hist, (int(pt_h[0]), int(pt_h[1])), radius, color, -1)
                        
                    if err_count > 0:
                        avg_err = total_err / err_count
                        
                    inlier_count = len(pts_h_256)
                    
    # Draw telemetry HUD console borders
    status_color = (0, 255, 0) if is_success else (0, 0, 255)
    cv2.rectangle(telemetry_panel, (0, 0), (255, 255), status_color, 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thickness = 1
    text_color = (220, 220, 220)
    label_color = (0, 255, 255)
    
    # Render metrics in telemetry console
    cv2.putText(telemetry_panel, "--- TELEMETRY HUD ---", (45, 25), font, 0.45, status_color, 1, cv2.LINE_AA)
    
    cv2.putText(telemetry_panel, "Detector: ", (15, 55), font, scale, label_color, thickness, cv2.LINE_AA)
    cv2.putText(telemetry_panel, f"{feature_type.upper()}", (95, 55), font, scale, text_color, thickness, cv2.LINE_AA)
    
    cv2.putText(telemetry_panel, "Proc Size: ", (15, 80), font, scale, label_color, thickness, cv2.LINE_AA)
    cv2.putText(telemetry_panel, f"{proc_size}x{proc_size}", (95, 80), font, scale, text_color, thickness, cv2.LINE_AA)
    
    cv2.putText(telemetry_panel, "Inliers: ", (15, 105), font, scale, label_color, thickness, cv2.LINE_AA)
    cv2.putText(telemetry_panel, f"{inlier_count}", (95, 105), font, scale, text_color, thickness, cv2.LINE_AA)
    
    cv2.putText(telemetry_panel, "Avg Epi Err:", (15, 130), font, scale, label_color, thickness, cv2.LINE_AA)
    avg_err_str = f"{avg_err:.4f} px" if avg_err > 0.0 else "N/A"
    cv2.putText(telemetry_panel, avg_err_str, (95, 130), font, scale, text_color, thickness, cv2.LINE_AA)
    
    hx_256 = hist_norm[0] * 256.0
    hy_256 = hist_norm[1] * 256.0
    px_256 = prev_norm[0] * 256.0
    py_256 = prev_norm[1] * 256.0
    cx_256 = curr_norm[0] * 256.0
    cy_256 = curr_norm[1] * 256.0
    
    cv2.putText(telemetry_panel, "Hist Coord:", (15, 160), font, scale, label_color, thickness, cv2.LINE_AA)
    cv2.putText(telemetry_panel, f"({hx_256:.1f}, {hy_256:.1f})", (105, 160), font, scale, text_color, thickness, cv2.LINE_AA)
    
    cv2.putText(telemetry_panel, "Prev Coord:", (15, 185), font, scale, label_color, thickness, cv2.LINE_AA)
    cv2.putText(telemetry_panel, f"({px_256:.1f}, {py_256:.1f})", (105, 185), font, scale, text_color, thickness, cv2.LINE_AA)
    
    cv2.putText(telemetry_panel, "Curr Coord:", (15, 210), font, scale, label_color, thickness, cv2.LINE_AA)
    cv2.putText(telemetry_panel, f"({cx_256:.1f}, {cy_256:.1f})", (105, 210), font, scale, text_color, thickness, cv2.LINE_AA)
    
    status_str = "SUCCESS" if is_success else ("HOVER" if sift_info is None else "FAILED")
    cv2.putText(telemetry_panel, "Status: ", (15, 240), font, scale, label_color, thickness, cv2.LINE_AA)
    cv2.putText(telemetry_panel, status_str, (95, 240), font, scale, status_color, 1, cv2.LINE_AA)
    
    # 6. Row 4: Pure Attention Masks & Expected Heatmap (256x256 each, column aligned)
    hist_mask_val = hist_mask[:, :, 0] if len(hist_mask.shape) == 3 else hist_mask
    prev_mask_val = prev_mask[:, :, 0] if len(prev_mask.shape) == 3 else prev_mask
    
    # Pure Hist attention mask - Red soft glow on black background
    hist_mask_panel = np.zeros((256, 256, 3), dtype=np.uint8)
    hist_mask_panel[:, :, 2] = (hist_mask_val * 255.0).astype(np.uint8)  # Red channel
    
    # Pure Prev attention mask - Blue soft glow on black background
    prev_mask_panel = np.zeros((256, 256, 3), dtype=np.uint8)
    prev_mask_panel[:, :, 0] = (prev_mask_val * 255.0).astype(np.uint8)  # Blue channel
    
    # Ground-truth expected output target heatmap, generated at 64x64 and scaled to 256x256 - Green soft glow on black background
    expected_heatmap_64 = generate_gaussian_heatmap(curr_norm, size=64, sigma=label_radius / 8.0)
    expected_heatmap_256 = cv2.resize(expected_heatmap_64, (256, 256), interpolation=cv2.INTER_LINEAR)
    if len(expected_heatmap_256.shape) == 3:
        expected_heatmap_val = expected_heatmap_256[:, :, 0]
    else:
        expected_heatmap_val = expected_heatmap_256
    expected_heatmap_panel = np.zeros((256, 256, 3), dtype=np.uint8)
    expected_heatmap_panel[:, :, 1] = (expected_heatmap_val * 255.0).astype(np.uint8)  # Green channel
    
    # Draw HUD labels
    draw_hud_label(h_color, "HIST CONTEXT + SOFT GLOW", (10, 240), (0, 0, 255))
    draw_hud_label(p_color, "PREV CONTEXT + SOFT GLOW", (10, 240), (255, 0, 0))
    draw_hud_label(c_color, "CURR CONTEXT (TARGET)", (10, 240), (0, 255, 0))
    
    sift_label = "GEOMETRIC CONNECTIONS (HIST -> PREV -> CURR)" if is_success else "GEOMETRIC FAILURE CONNECTIONS"
    sift_label_color = (255, 255, 0) if is_success else (0, 128, 255)
    draw_hud_label(s_color, sift_label, (10, 240), sift_label_color)
    
    draw_hud_label(epipoles_on_prev, "EPIPOLAR LINES: HIST ON PREV", (10, 240), (255, 0, 255) if is_success else (0, 0, 255))
    draw_hud_label(epipoles_on_hist, "EPIPOLAR LINES: PREV ON HIST", (10, 240), (255, 0, 255) if is_success else (0, 0, 255))
    
    draw_hud_label(hist_mask_panel, "HIST ATTENTION MASK (RED)", (10, 240), (0, 0, 255))
    draw_hud_label(prev_mask_panel, "PREV ATTENTION MASK (BLUE)", (10, 240), (255, 0, 0))
    draw_hud_label(expected_heatmap_panel, "EXPECTED HEATMAP (GREEN)", (10, 240), (0, 255, 0))
    
    # 7. Assemble Grid (Symmetric 4x3 Grid)
    row1 = np.hstack([h_color, p_color, c_color])  # Shape: (256, 768, 3)
    row2 = s_color                                # Shape: (256, 768, 3)
    row3 = np.hstack([epipoles_on_hist, epipoles_on_prev, telemetry_panel]) # Shape: (256, 768, 3)
    row4 = np.hstack([hist_mask_panel, prev_mask_panel, expected_heatmap_panel]) # Shape: (256, 768, 3)
    dashboard = np.vstack([row1, row2, row3, row4])     # Shape: (1024, 768, 3)
    
    # 8. Add Top HUD Dashboard header bar (768px wide)
    header_bar = np.zeros((35, 768, 3), dtype=np.uint8)
    
    if is_success or sift_info is None:
        title_text = "VIDEO DATASET GENERATOR - PREVIEW HUD"
        title_color = (0, 255, 255)
    else:
        inliers_count = len(sift_info.get("inliers", []))
        reason = sift_info.get("reason", "unknown_failure")
        title_text = f"MATCH FAILURE: Found {inliers_count} inliers ({reason})"
        title_color = (0, 0, 255)
        
    cv2.putText(header_bar, title_text, (10, 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, title_color, 1, cv2.LINE_AA)
    cv2.putText(header_bar, "[SPACE]: Next  |  [ENTER]: Auto-run  |  [ESC/Q]: Exit", (460, 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    
    final_output = np.vstack([header_bar, dashboard])
    return final_output

# =====================================================================
# Pipelines: Interactive Visualization and Two-Stage Production
# =====================================================================

def visualize_pipeline(args, video_paths, asift_matcher):
    print("\n=== ENTERING HUD PREVIEW MODE ===")
    print("Rendering generated sequences in real-time. No files will be exported to disk.")
    print("Controls: Press [SPACE] for next single step, [ENTER] to auto-run until success, [ESC/Q] to exit.\n")
    sys.stdout.flush()
    try:
        cv2.namedWindow("Video Dataset Generator Debugger", cv2.WINDOW_AUTOSIZE)
    except cv2.error as e:
        print(f"Warning: Could not initialize visual window ({e}).")
        print("If you are running in a headless environment, please run without '-v' / '--visualize'.")
        sys.exit(1)
        
    # Pre-allocate and shuffle decisions to guarantee exact ratio and perfect random distribution (i.i.d.)
    num_hover_target = int(args.batch_size * args.hover_prob)
    num_trans_target = args.batch_size - num_hover_target
    
    batch_decisions = [True] * num_hover_target + [False] * num_trans_target
    random.shuffle(batch_decisions)
    
    auto_run_until_success = False
    attempt_count = 0
    sample_count = 0
    
    while sample_count < args.num_of_samples:
        attempt_count += 1
        is_hover = batch_decisions[sample_count % args.batch_size]
        
        random_video = random.choice(video_paths)
        cap = cv2.VideoCapture(random_video)
        
        # Robust corrupted video logging as requested by user
        if not cap.isOpened():
            print(f"[ERROR] Corrupted video file (moov atom not found / failed to open): {os.path.abspath(random_video)}")
            sys.stdout.flush()
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 200:
            print(f"[ERROR] Corrupted or incomplete video file (too few frames / moov atom failure): {os.path.abspath(random_video)}")
            sys.stdout.flush()
            cap.release()
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
            
        gap_k = random.randint(int(0.9 * fps), int(1.1 * fps))
        max_start = total_frames - int(fps * 1.5)
        start_frame_idx = random.randint(0, max_start)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        ret, frame_hist_raw = cap.read()
        if not ret or frame_hist_raw is None:
            cap.release()
            continue
            
        f_hist_full = to_grayscale(frame_hist_raw)
        f_hist_256 = cv2.resize(f_hist_full, (256, 256), interpolation=cv2.INTER_AREA)
        
        if is_hover:
            cap.release()
            sift = cv2.SIFT_create()
            kp, _ = sift.detectAndCompute(f_hist_256, None)
            if not kp:
                continue
            target_kp = random.choice(kp)
            target_coords = [target_kp.pt[0] / 256.0, target_kp.pt[1] / 256.0]
            f_prev_256, prev_coords = simulate_hover_jitter(f_hist_256, target_coords)
            f_curr_256, curr_coords = simulate_hover_jitter(f_hist_256, target_coords)
            hist_coords = target_coords
            sift_match_debug = None
        else:
            f_hist_proc = cv2.resize(f_hist_full, (args.proc_size, args.proc_size), interpolation=cv2.INTER_AREA)
            
            for _ in range(gap_k - 1):
                cap.grab()
            ret, frame_prev_raw = cap.read()
            if not ret or frame_prev_raw is None:
                cap.release()
                continue
            f_prev_full = to_grayscale(frame_prev_raw)
            f_prev_256 = cv2.resize(f_prev_full, (256, 256), interpolation=cv2.INTER_AREA)
            f_prev_proc = cv2.resize(f_prev_full, (args.proc_size, args.proc_size), interpolation=cv2.INTER_AREA)
            
            gap_d = 2
            for _ in range(gap_d - 1):
                cap.grab()
            ret, frame_curr_raw = cap.read()
            cap.release()
            if not ret or frame_curr_raw is None:
                continue
            f_curr_full = to_grayscale(frame_curr_raw)
            f_curr_256 = cv2.resize(f_curr_full, (256, 256), interpolation=cv2.INTER_AREA)
            f_curr_proc = cv2.resize(f_curr_full, (args.proc_size, args.proc_size), interpolation=cv2.INTER_AREA)
            
            match_res = match_features_triplet(
                f_hist_proc, f_prev_proc, f_curr_proc,
                ratio=args.ratio, min_inliers=args.min_inliers,
                feature_type=args.feature_type, asift_matcher=asift_matcher,
                ransac_thresh=args.ransac_thresh,
                min_motion_pc=args.min_motion_pc,
                min_motion_hp=args.min_motion_hp,
                min_texture_std=args.min_texture_std,
                proc_size=args.proc_size,
                min_ncc=args.min_ncc
            )
            
            if match_res.get("status") == "success":
                selected_path = None
                for path in match_res["paths"]:
                    if is_path_in_inner_2_3(path):
                        h_pt = path["hist"]
                        p_pt = path["prev"]
                        c_pt = path["curr"]
                        dist_pc = np.sqrt((c_pt[0] - p_pt[0])**2 + (c_pt[1] - p_pt[1])**2) * 256.0
                        dist_hp = np.sqrt((p_pt[0] - h_pt[0])**2 + (p_pt[1] - h_pt[1])**2) * 256.0
                        if dist_pc >= args.target_min_motion_pc and dist_hp >= args.target_min_motion_hp:
                            selected_path = path
                            break
                if selected_path is not None:
                    hist_coords = selected_path["hist"]
                    prev_coords = selected_path["prev"]
                    curr_coords = selected_path["curr"]
                else:
                    match_res["status"] = "failed"
                    match_res["reason"] = "no_inlier_in_inner_2_3_or_motion_failed"
                    hist_coords = [0.5, 0.5]
                    prev_coords = [0.5, 0.5]
                    curr_coords = [0.5, 0.5]
            else:
                hist_coords = [0.5, 0.5]
                prev_coords = [0.5, 0.5]
                curr_coords = [0.5, 0.5]
                
            sift_match_debug = match_res
            
        hist_mask = generate_attention_mask(
            hist_coords, size=256, mask_type=args.mask_type,
            radius=args.hist_radius, sigma=args.mask_sigma
        )
        prev_mask = generate_attention_mask(
            prev_coords, size=256, mask_type=args.mask_type,
            radius=args.prev_radius, sigma=args.mask_sigma
        )
        
        is_success = (sift_match_debug is not None and sift_match_debug.get("status") == "success") or is_hover
        
        dashboard = render_dashboard(
            f_hist_256, f_prev_256, f_curr_256,
            hist_mask, prev_mask,
            hist_coords, prev_coords, curr_coords,
            sift_info=sift_match_debug,
            proc_size=args.proc_size,
            feature_type=args.feature_type,
            label_radius=args.label_radius
        )
        try:
            cv2.imshow("Video Dataset Generator Debugger", dashboard)
            if is_success and auto_run_until_success:
                auto_run_until_success = False
            delay = 100 if auto_run_until_success else 0
            key = cv2.waitKey(delay) & 0xFF
            
            if key == 13 or key == 10:
                auto_run_until_success = True
            elif key == 32:
                auto_run_until_success = False
            elif key == 27 or key == ord('q'):
                cv2.destroyAllWindows()
                print("\nHUD Preview Mode exited by user.")
                sys.exit(0)
        except cv2.error as e:
            print(f"\nGUI Error: Could not render visualization window ({e}).")
            sys.exit(1)
            
        if is_success:
            sample_count += 1
            print(f"[Sample {sample_count} | Attempt {attempt_count}] Rendered success (Hover: {is_hover})")
            sys.stdout.flush()
            attempt_count = 0
        else:
            print(f"[Sample {sample_count + 1} | Attempt {attempt_count}] Match Failed ({args.feature_type}): {sift_match_debug.get('reason')} - rendering connections.")
            sys.stdout.flush()


def run_two_stage_pipeline(args, video_paths, asift_matcher):
    print("\n==========================================================")
    print("=== ENTERING TWO-STAGE HIGH-THROUGHPUT GENERATION PIPE ===")
    print("==========================================================\n")
    sys.stdout.flush()
    
    # -------------------------------------------------------------------------
    # STAGE 1: Sequential Extraction & Multi-Target Path Gathering
    # -------------------------------------------------------------------------
    tmp_dir = os.path.join(args.output_dir, "tmp_raw_extracted")
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Map video total frames to predict total project time accurately
    video_lengths = {}
    video_fps_map = {}
    
    print("Pre-mapping video durations for hyper-accurate time prediction HUD...")
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30.0
            video_lengths[path] = frames
            video_fps_map[path] = fps
        cap.release()
    print(f"Mapped {len(video_lengths)} active videos.\n")
    sys.stdout.flush()
    
    # Find already completed videos in tmp_raw_extracted using JSON tracking files
    processed_files = os.listdir(tmp_dir)
    processed_hashes = set()
    for filename in processed_files:
        if filename.startswith("processed_") and filename.endswith(".json"):
            parts = filename[:-5].split("_")
            if len(parts) >= 3:
                processed_hashes.add(parts[-1])
                
    remaining_videos = []
    for path in video_paths:
        v_hash = hashlib.md5(path.encode()).hexdigest()[:8]
        if v_hash not in processed_hashes:
            remaining_videos.append(path)
            
    print(f"Resume HUD Status: {len(video_paths) - len(remaining_videos)}/{len(video_paths)} videos already completed.")
    print(f"Remaining videos to process: {len(remaining_videos)}\n")
    sys.stdout.flush()
    
    # Start loop over remaining videos
    for idx, path in enumerate(remaining_videos):
        video_name = os.path.basename(path)
        v_hash = hashlib.md5(path.encode()).hexdigest()[:8]
        metadata_filename = f"processed_{os.path.splitext(video_name)[0]}_{v_hash}.json"
        metadata_path = os.path.join(tmp_dir, metadata_filename)
        
        print(f"\n[STAGING STAGE 1] Starting sequential extraction on Video {idx+1}/{len(remaining_videos)}: {video_name}")
        sys.stdout.flush()
        
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[ERROR] Corrupted video file (failed to open): {os.path.abspath(path)}")
            # Save empty placeholder JSON so we skip this corrupted file in future runs
            with open(metadata_path, "w") as f:
                json.dump({"video_path": path, "sample_count": 0, "status": "corrupted"}, f)
            continue
            
        total_frames = video_lengths.get(path, 200)
        fps = video_fps_map.get(path, 30.0)
        
        if total_frames < 200:
            print(f"[ERROR] Corrupted or incomplete video file (too few frames): {os.path.abspath(path)}")
            cap.release()
            with open(metadata_path, "w") as f:
                json.dump({"video_path": path, "sample_count": 0, "status": "too_few_frames"}, f)
            continue
            
        # Step size in frames (default step is 1 second)
        frame_step = int(round(args.temporal_step_seconds * fps))
        gap_k = int(round(fps))  # ~1 second gap between hist and prev
        
        video_sample_count = 0
        consecutive_failures = 0
        current_frame_idx = 0
        max_start = total_frames - int(fps * 1.5)
        
        t_start_video = time.time()
        triplets_processed = 0
        successful_frames = 0
        
        while current_frame_idx <= max_start:
            # Initialize loop variables to None for safe deletion
            f_hist_full = f_hist_256 = f_hist_proc = None
            f_prev_full = f_prev_256 = f_prev_proc = None
            f_curr_full = f_curr_256 = f_curr_proc = None
            frame_hist_raw = frame_prev_raw = frame_curr_raw = None
            match_res = None
            
            # Seek and read hist frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame_hist_raw = cap.read()
            if not ret or frame_hist_raw is None:
                break
                
            # Seek and read prev frame (after gap_k frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx + gap_k)
            ret_p, frame_prev_raw = cap.read()
            if not ret_p or frame_prev_raw is None:
                break
                
            # Seek and read curr frame (after another 2 frames gap)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx + gap_k + 2)
            ret_c, frame_curr_raw = cap.read()
            if not ret_c or frame_curr_raw is None:
                break
                
            triplets_processed += 1
            
            # Grayscale conversions
            f_hist_full = to_grayscale(frame_hist_raw)
            f_hist_256 = cv2.resize(f_hist_full, (256, 256), interpolation=cv2.INTER_AREA)
            f_hist_proc = cv2.resize(f_hist_full, (args.proc_size, args.proc_size), interpolation=cv2.INTER_AREA)
            
            f_prev_full = to_grayscale(frame_prev_raw)
            f_prev_256 = cv2.resize(f_prev_full, (256, 256), interpolation=cv2.INTER_AREA)
            f_prev_proc = cv2.resize(f_prev_full, (args.proc_size, args.proc_size), interpolation=cv2.INTER_AREA)
            
            f_curr_full = to_grayscale(frame_curr_raw)
            f_curr_256 = cv2.resize(f_curr_full, (256, 256), interpolation=cv2.INTER_AREA)
            f_curr_proc = cv2.resize(f_curr_full, (args.proc_size, args.proc_size), interpolation=cv2.INTER_AREA)
            
            # Match SIFT/ASIFT/SURF keypoints
            match_res = match_features_triplet(
                f_hist_proc, f_prev_proc, f_curr_proc, 
                ratio=args.ratio, min_inliers=args.min_inliers,
                feature_type=args.feature_type, asift_matcher=asift_matcher,
                ransac_thresh=args.ransac_thresh,
                min_motion_pc=args.min_motion_pc,
                min_motion_hp=args.min_motion_hp,
                min_texture_std=args.min_texture_std,
                proc_size=args.proc_size,
                min_ncc=args.min_ncc,
                keep_debug_info=False
            )
            
            frame_yield = 0
            
            if match_res.get("status") == "success":
                consecutive_failures = 0
                
                # Multiplexing: Extract ALL valid paths in this single frame read!
                for path_candidate in match_res["paths"]:
                    if is_path_in_inner_2_3(path_candidate):
                        # Calculate motion vectors for this specific path candidate
                        h_pt = path_candidate["hist"]
                        p_pt = path_candidate["prev"]
                        c_pt = path_candidate["curr"]
                        
                        dist_pc = np.sqrt((c_pt[0] - p_pt[0])**2 + (c_pt[1] - p_pt[1])**2) * 256.0
                        dist_hp = np.sqrt((p_pt[0] - h_pt[0])**2 + (p_pt[1] - h_pt[1])**2) * 256.0
                        
                        if dist_pc >= args.target_min_motion_pc and dist_hp >= args.target_min_motion_hp:
                            # Valid moving target path inside inner 2/3! Extract it!
                            hist_mask = generate_attention_mask(
                                h_pt, size=256, mask_type=args.mask_type,
                                radius=args.hist_radius, sigma=args.mask_sigma
                            )
                            prev_mask = generate_attention_mask(
                                p_pt, size=256, mask_type=args.mask_type,
                                radius=args.prev_radius, sigma=args.mask_sigma
                            )
                            zeros_mask = np.zeros((256, 256, 1), dtype=np.float32)
                            
                            # Convert to target training formats (Channel 0 = Grayscale, Channel 1 = Mask)
                            f_hist_norm = f_hist_256.astype(np.float32) / 255.0
                            f_prev_norm = f_prev_256.astype(np.float32) / 255.0
                            f_curr_norm = f_curr_256.astype(np.float32) / 255.0
                            
                            # Stack along last axis (channel axis) to produce shape (256, 256, 2)
                            hist_frame = np.stack([f_hist_norm, hist_mask[:, :, 0]], axis=-1)
                            prev_frame = np.stack([f_prev_norm, prev_mask[:, :, 0]], axis=-1)
                            curr_frame = np.stack([f_curr_norm, zeros_mask[:, :, 0]], axis=-1)
                            
                            target_heatmap = generate_gaussian_heatmap(c_pt, size=64, sigma=args.label_radius / 8.0)
                            
                            # Store target sample dict
                            sample_data = {
                                "hist_frame": hist_frame,
                                "prev_frame": prev_frame,
                                "curr_frame": curr_frame,
                                "target_heatmap": target_heatmap,
                                "debug_coords": {
                                    "hist": h_pt,
                                    "prev": p_pt,
                                    "curr": c_pt
                                }
                            }
                            # Save individual sample directly to disk to prevent RAM accumulation
                            sample_filename = f"sample_{v_hash}_{video_sample_count}.pkl"
                            sample_path = os.path.join(tmp_dir, sample_filename)
                            with open(sample_path, "wb") as f_out:
                                pickle.dump(sample_data, f_out)
                            video_sample_count += 1
                            frame_yield += 1
                            
                if frame_yield > 0:
                    successful_frames += 1
            else:
                consecutive_failures += 1
                
            # Advance to next temporal step (chronological sweep)
            current_frame_idx += frame_step
            
            # Print live high-fidelity stdout HUD console
            elapsed = time.time() - t_start_video
            rate = triplets_processed / elapsed if elapsed > 0 else 0.0
            
            # Calculate remaining triplets in current video
            remaining_triplets = max(0, (max_start - current_frame_idx) // frame_step)
            eta_current_sec = remaining_triplets / rate if rate > 0 else 0.0
            
            # Calculate remaining triplets overall across all videos
            remaining_overall_triplets = remaining_triplets
            for remaining_path in remaining_videos[idx+1:]:
                frames_rem = video_lengths.get(remaining_path, 200)
                fps_rem = video_fps_map.get(remaining_path, 30.0)
                max_start_rem = frames_rem - int(fps_rem * 1.5)
                frame_step_rem = int(round(args.temporal_step_seconds * fps_rem))
                remaining_overall_triplets += max(0, max_start_rem // frame_step_rem)
                
            eta_overall_sec = remaining_overall_triplets / rate if rate > 0 else 0.0
            
            # Format times
            percent_complete = int((current_frame_idx / max_start) * 100) if max_start > 0 else 0
            percent_complete = min(100, max(0, percent_complete))
            
            current_time_str = time.strftime('%M:%S', time.gmtime(int(current_frame_idx / fps)))
            total_time_str = time.strftime('%M:%S', time.gmtime(int(total_frames / fps)))
            
            eta_curr_str = f"{int(eta_current_sec // 60)}m {int(eta_current_sec % 60)}s" if eta_current_sec > 0 else "N/A"
            eta_over_str = f"{int(eta_overall_sec // 60)}m {int(eta_overall_sec % 60)}s" if eta_overall_sec > 0 else "N/A"
            
            sys.stdout.write(
                f"\r[STAGING STAGE 1] Progress: {percent_complete}% ({current_time_str}/{total_time_str}) | "
                f"Yield: {successful_frames} frames ({video_sample_count} samples) | "
                f"Rate: {rate:.1f} Hz | ETA Video: {eta_curr_str} | ETA Total: {eta_over_str}   "
            )
            sys.stdout.flush()
            
            # Explicit memory cleanup of heavy matrices in loop
            del f_hist_full, f_hist_256, f_hist_proc
            del f_prev_full, f_prev_256, f_prev_proc
            del f_curr_full, f_curr_256, f_curr_proc
            del frame_hist_raw, frame_prev_raw, frame_curr_raw
            del match_res
            gc.collect()
            
            # Safeguard: skip static/bad/untrackable video regions
            if consecutive_failures >= args.max_consecutive_failures:
                print(f"\n[WARNING] Too many consecutive failures ({consecutive_failures}/{args.max_consecutive_failures}) due to low-texture / static scene. Skipping remainder of video.")
                break
                
        cap.release()
        
        # Write JSON tracking metadata indicating successful completion of the video
        with open(metadata_path, "w") as f:
            json.dump({
                "video_path": path,
                "sample_count": video_sample_count,
                "status": "success",
                "v_hash": v_hash
            }, f)
            
        print(f"\n[SUCCESS] Completed Video {idx+1}/{len(remaining_videos)}. Yielded {video_sample_count} samples. Saved completion tracking -> {metadata_path}\n")
        sys.stdout.flush()
        
        # Explicit garbage collection after each video
        gc.collect()
        
    # -------------------------------------------------------------------------
    # STAGE 2: Balanced, Shuffled I.I.D. Training Compilation
    # -------------------------------------------------------------------------
    print("\n==========================================================")
    print("=== STAGE 2: BALANCED I.I.D. TRAINING DATASET COMPILATION ===")
    print("==========================================================\n")
    sys.stdout.flush()
    
    # Find all JSON completion metadata files from the temporary staging folder
    raw_jsons = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.startswith("processed_") and f.endswith(".json")]
    
    video_yields = {}
    
    # Gather the filenames of all individual sample pickles
    sample_files = [f for f in os.listdir(tmp_dir) if f.startswith("sample_") and f.endswith(".pkl")]
    total_samples = len(sample_files)
    
    print("Consolidating all temporary pre-processed sample records...")
    for json_file in raw_jsons:
        try:
            with open(json_file, "r") as f:
                meta = json.load(f)
                video_yields[os.path.basename(json_file)] = meta.get("sample_count", 0)
        except Exception as e:
            print(f"Warning: Could not read temporary json file '{json_file}' ({e}). Skipping.")
            
    print(f"Total compiled samples extracted from all videos: {total_samples}")
    for v_name, v_yield in video_yields.items():
        print(f" - {v_name}: {v_yield} samples")
    print("")
    sys.stdout.flush()
    
    if total_samples == 0:
        print("[ERROR] No valid dataset samples were extracted in Stage 1. Please ensure videos are present and match settings.")
        sys.exit(1)
        
    # Shuffle all collected sample filenames globally to guarantee perfect i.i.d. training balance
    print("Shuffling all collected sample filenames to guarantee perfect i.i.d. training balance...")
    random.shuffle(sample_files)
    print("Shuffling complete.\n")
    sys.stdout.flush()
    
    # Package into training batch files matching tracker_model expectation exactly
    num_batches = int(np.ceil(total_samples / args.batch_size))
    print(f"Packaging {total_samples} samples into {num_batches} batches of size {args.batch_size}...")
    sys.stdout.flush()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for b in range(num_batches):
        batch_slice = sample_files[b * args.batch_size : (b + 1) * args.batch_size]
        
        # If the last batch is smaller, pad it or wrap around to keep exactly batch_size
        if len(batch_slice) < args.batch_size:
            pad_needed = args.batch_size - len(batch_slice)
            for i in range(pad_needed):
                batch_slice.append(sample_files[i % len(sample_files)])
                
        hist_frames_batch = []
        prev_frames_batch = []
        curr_frames_batch = []
        target_heatmaps_batch = []
        
        hist_coords_batch = []
        prev_coords_batch = []
        curr_coords_batch = []
        
        for sample_filename in batch_slice:
            sample_path = os.path.join(tmp_dir, sample_filename)
            try:
                with open(sample_path, "rb") as f_in:
                    sample = pickle.load(f_in)
                    
                hist_frames_batch.append(sample["hist_frame"])
                prev_frames_batch.append(sample["prev_frame"])
                curr_frames_batch.append(sample["curr_frame"])
                target_heatmaps_batch.append(sample["target_heatmap"])
                
                hist_coords_batch.append(sample["debug_coords"]["hist"])
                prev_coords_batch.append(sample["debug_coords"]["prev"])
                curr_coords_batch.append(sample["debug_coords"]["curr"])
            except Exception as e:
                print(f"[ERROR] Failed to load sample file '{sample_path}': {e}")
                sys.exit(1)
                
        batch_data = {
            "inputs": [
                np.array(hist_frames_batch, dtype=np.float32),
                np.array(prev_frames_batch, dtype=np.float32),
                np.array(curr_frames_batch, dtype=np.float32)
            ],
            "targets": np.array(target_heatmaps_batch, dtype=np.float32),
            "debug_coords": {
                "hist": np.array(hist_coords_batch, dtype=np.float32),
                "prev": np.array(prev_coords_batch, dtype=np.float32),
                "curr": np.array(curr_coords_batch, dtype=np.float32)
            }
        }
        
        output_file = os.path.join(args.output_dir, f"video_dataset_{b}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(batch_data, f)
            
        print(f"Generated and saved final batch {b+1}/{num_batches} -> {output_file}")
        sys.stdout.flush()
        
        # Explicit memory cleanup of batch lists and gc trigger
        del batch_data, hist_frames_batch, prev_frames_batch, curr_frames_batch, target_heatmaps_batch
        del hist_coords_batch, prev_coords_batch, curr_coords_batch
        gc.collect()
        
    print("\n==========================================================")
    print("=== HIGH-THROUGHPUT TWO-STAGE DATASET GENERATION COMPLETE ===")
    print("==========================================================\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Generates training datasets for tracker_model directly from real driving/drone videos using ASIFT RANSAC."
    )
    parser.add_argument(
        "videos_dir",
        help="Path to the directory containing video files to crawl."
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="video_dataset",
        help="Output directory path where pickle batch files will be saved (default: video_dataset)."
    )
    parser.add_argument(
        "-v", "--visualize",
        action="store_true",
        help="Enables interactive high-fidelity preview mode rendering SIFT matches in real-time."
    )
    parser.add_argument(
        "--num_of_samples",
        type=int,
        default=16384,
        help="Total number of sequence samples to generate (default: 16384)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size per saved pickle file (default: 256)."
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.85,
        help="Lowe's ratio threshold for filtering SIFT matches. Eased for perspective matching (default: 0.85)."
    )
    parser.add_argument(
        "--min_inliers",
        type=int,
        default=6,
        help="Minimum number of verified RANSAC inliers to accept a triplet. Eased for perspective matching (default: 6)."
    )
    parser.add_argument(
        "--hover_prob",
        type=float,
        default=0.05,
        help="Probability of selecting stationary hovering with gimbal camera shake (default: 0.05)."
    )
    parser.add_argument(
        "--mask_type",
        default="gaussian",
        choices=["circular", "gaussian"],
        help="Type of attention mask generated in Channel 1 (default: gaussian)."
    )
    parser.add_argument(
        "--mask_sigma",
        type=float,
        default=15.0,
        help="Standard deviation (sigma) of the Gaussian soft mask in pixels (default: 15.0)."
    )
    parser.add_argument(
        "--hist_radius",
        type=int,
        default=128,
        help="Radius of the historical circular mask in pixels (default: 128)."
    )
    parser.add_argument(
        "--prev_radius",
        type=int,
        default=50,
        help="Radius of the previous circular mask in pixels (default: 50)."
    )
    parser.add_argument(
        "--label_radius",
        type=int,
        default=32,
        help="Radius of the target label expected heatmap in pixels (default: 32)."
    )
    parser.add_argument(
        "--feature_type",
        default="asift",
        choices=["asift", "sift", "surf"],
        help="Feature matching algorithm. ASIFT is recommended for perspective dilation (default: asift)."
    )
    parser.add_argument(
        "--proc_size",
        type=int,
        default=800,
        help="Image resolution for feature extraction, keypoint matching, and RANSAC verification (default: 800)."
    )
    parser.add_argument(
        "--ransac_thresh",
        type=float,
        default=5.0,
        help="RANSAC reprojection error threshold in pixels (default: 5.0)."
    )
    parser.add_argument(
        "--min_motion_pc",
        type=float,
        default=0.0,
        help="Minimum motion velocity in pixels between prev and curr frames for keypoints (default: 0.0)."
    )
    parser.add_argument(
        "--min_motion_hp",
        type=float,
        default=0.0,
        help="Minimum motion velocity in pixels between hist and prev frames for keypoints (default: 0.0)."
    )
    parser.add_argument(
        "--target_min_motion_pc",
        type=float,
        default=1.0,
        help="Minimum required motion velocity in pixels for the chosen target between prev and curr (default: 1.0)."
    )
    parser.add_argument(
        "--target_min_motion_hp",
        type=float,
        default=3.0,
        help="Minimum required motion velocity in pixels for the chosen target between hist and prev (default: 3.0)."
    )
    parser.add_argument(
        "--min_texture_std",
        type=float,
        default=3.0,
        help="Minimum local patch standard deviation to accept keypoints in low-contrast areas (default: 3.0)."
    )
    parser.add_argument(
        "--min_ncc",
        type=float,
        default=0.75,
        help="Minimum Normalized Cross-Correlation (NCC) patch similarity to accept matches (default: 0.75)."
    )
    parser.add_argument(
        "--temporal_step_seconds",
        type=float,
        default=1.0,
        help="Constant time gap in seconds between chronological seeks in Stage 1 sequential mode (default: 1.0)."
    )
    parser.add_argument(
        "--max_consecutive_failures",
        type=int,
        default=10,
        help="Maximum consecutive match failures allowed in a single video before skipping it in Stage 1 (default: 10)."
    )
    
    # Load and apply pipeline_config.json defaults dynamically
    config_path = os.path.join(os.path.dirname(__file__), "pipeline_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                valid_keys = {action.dest for action in parser._actions if action.dest != 'help'}
                filtered_config = {k: v for k, v in config.items() if k in valid_keys}
                parser.set_defaults(**filtered_config)
        except Exception as e:
            print(f"Warning: Could not load pipeline_config.json ({e}). Using CLI defaults.")
            
    args = parser.parse_args()
    
    if not os.path.isdir(args.videos_dir):
        print(f"Error: Directory '{args.videos_dir}' does not exist or is not a directory.")
        sys.exit(1)
        
    print("Crawling for video files...")
    video_paths = find_video_files(args.videos_dir)
    if not video_paths:
        print(f"Error: No video files found in directory: {args.videos_dir}")
        sys.exit(1)
        
    print(f"Found {len(video_paths)} valid videos.")
    
    # Initialize the ASIFT matcher if selected
    asift_matcher = None
    if args.feature_type == "asift":
        asift_matcher = ASIFTMatcher(feature_type="sift")
        
    if args.visualize:
        visualize_pipeline(args, video_paths, asift_matcher)
    else:
        run_two_stage_pipeline(args, video_paths, asift_matcher)


if __name__ == "__main__":
    main()

