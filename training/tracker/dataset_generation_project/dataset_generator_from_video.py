#!/usr/bin/env python3
import os
import sys
import random
import pickle
import argparse
import numpy as np
import cv2

# =====================================================================
# Heatmap and Attention Masking Helpers (Matching tracker_model3 exactly)
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
                           ransac_thresh=5.0, min_motion_pc=1.0, min_motion_hp=3.0, min_texture_std=3.0, proc_size=800, min_ncc=0.75):
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
            "kp_hist": kp_hist if kp_hist else [],
            "kp_prev": kp_prev if kp_prev else [],
            "kp_curr": kp_curr if kp_curr else [],
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

def render_dashboard(f_hist_256, f_prev_256, f_curr_256, hist_mask, prev_mask, hist_norm, prev_norm, curr_norm, sift_info=None, proc_size=800, feature_type='asift'):
    """
    Renders a stunning 3x3 widescreen grid dashboard containing full frames overlaid with highly transparent
    colored attention masks, green target indicators, and inter-image keypoint matching lines.
    Additionally displays two real-time epipolar diagnostic panels and a telemetry console (Row 3).
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
    
    # Highly transparent blend (alpha = 0.25, beta = 1.0)
    cv2.addWeighted(h_mask_bgr, 0.25, h_color, 1.0, 0, h_color)
    cv2.addWeighted(p_mask_bgr, 0.25, p_color, 1.0, 0, p_color)
    
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
    
    # 6. Draw HUD labels
    draw_hud_label(h_color, "HIST CONTEXT + SOFT GLOW", (10, 240), (0, 0, 255))
    draw_hud_label(p_color, "PREV CONTEXT + SOFT GLOW", (10, 240), (255, 0, 0))
    draw_hud_label(c_color, "CURR CONTEXT (TARGET)", (10, 240), (0, 255, 0))
    
    sift_label = "GEOMETRIC CONNECTIONS (HIST -> PREV -> CURR)" if is_success else "GEOMETRIC FAILURE CONNECTIONS"
    sift_label_color = (255, 255, 0) if is_success else (0, 128, 255)
    draw_hud_label(s_color, sift_label, (10, 240), sift_label_color)
    
    draw_hud_label(epipoles_on_prev, "EPIPOLAR LINES: HIST ON PREV", (10, 240), (255, 0, 255) if is_success else (0, 0, 255))
    draw_hud_label(epipoles_on_hist, "EPIPOLAR LINES: PREV ON HIST", (10, 240), (255, 0, 255) if is_success else (0, 0, 255))
    
    # 7. Assemble Grid (Symmetric 3x3 Grid)
    row1 = np.hstack([h_color, p_color, c_color])  # Shape: (256, 768, 3)
    row2 = s_color                                # Shape: (256, 768, 3)
    row3 = np.hstack([epipoles_on_prev, epipoles_on_hist, telemetry_panel]) # Shape: (256, 768, 3)
    dashboard = np.vstack([row1, row2, row3])     # Shape: (768, 768, 3)
    
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
# Main Dataset Generation Loop
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generates training datasets for tracker_model3 directly from real driving/drone videos using ASIFT RANSAC."
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
    
    if args.visualize:
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
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\nTarget output folder: '{args.output_dir}'")
        
    num_batches = int(np.ceil(args.num_of_samples / args.batch_size))
    total_samples = num_batches * args.batch_size
    
    if not args.visualize:
        print(f"Generating {total_samples} samples ({num_batches} batches of size {args.batch_size}).")
        
    sample_count = 0
    
    # State trackers for visual HUD auto-running and counting
    auto_run_until_success = False
    attempt_count = 0
    
    # Initialize the ASIFT matcher if selected
    asift_matcher = None
    if args.feature_type == "asift":
        asift_matcher = ASIFTMatcher(feature_type="sift")
        
    for b in range(num_batches):
        if args.visualize and sample_count >= args.num_of_samples:
            break
            
        hist_frames_batch = []
        prev_frames_batch = []
        curr_frames_batch = []
        target_heatmaps_batch = []
        
        hist_coords_batch = []
        prev_coords_batch = []
        curr_coords_batch = []
        
        # Pre-allocate and shuffle decisions to guarantee exact ratio and perfect random distribution (i.i.d.)
        num_hover_target = int(args.batch_size * args.hover_prob)
        num_trans_target = args.batch_size - num_hover_target
        
        batch_decisions = [True] * num_hover_target + [False] * num_trans_target
        random.shuffle(batch_decisions)
        
        while len(hist_frames_batch) < args.batch_size:
            if args.visualize and sample_count >= args.num_of_samples:
                break
                
            attempt_count += 1
            is_hover = batch_decisions[len(hist_frames_batch)]
            
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
                
            # Time gap of approx. 1 second between hist and prev (0.9 to 1.1s randomized window for training generalizability)
            gap_k = random.randint(int(0.9 * fps), int(1.1 * fps))
            
            # Define maximum starting index to prevent running out of frames (at least 1.5 seconds)
            max_start = total_frames - int(fps * 1.5)
            start_frame_idx = random.randint(0, max_start)
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
            ret, frame_hist_raw = cap.read()
            if not ret or frame_hist_raw is None:
                cap.release()
                continue
                
            # Grayscale conversions
            f_hist_full = to_grayscale(frame_hist_raw)
            f_hist_256 = cv2.resize(f_hist_full, (256, 256), interpolation=cv2.INTER_AREA)
            
            if is_hover:
                # =========================================================
                # Stationary Hover Scenario (SIFT on 1 frame + Jitter)
                # =========================================================
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
                # =========================================================
                # Real Drone / Driving Camera Translation (Consecutive Seek)
                # =========================================================
                # Real Drone / Driving Camera Translation (Consecutive Seek)
                # gap_k is pre-computed dynamically above to represent approx. 1 second based on video FPS
                # Create processing-resolution grayscale frame for hist frame
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
                
                # Skip to current frame (very small step)
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
                
                # Match SIFT/ASIFT/SURF keypoints across the three branches (using customizable thresholds in proc_size space)
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
                
                # If in production dataset generation, skip failed SIFT triplets
                if not args.visualize:
                    if match_res is None or match_res.get("status") == "failed":
                        continue
                        
                # Pick the mathematically best landmark trajectory (lowest epipolar line error)
                # that is strictly located within the inner 2/3 center region to avoid lens distortion
                # and satisfies our tracking target motion requirements
                if match_res.get("status") == "success":
                    selected_path = None
                    for path in match_res["paths"]:
                        if is_path_in_inner_2_3(path):
                            # Calculate motion vectors for this specific path candidate in 256x256 space
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
                        # No inliers fell into the inner 2/3 region and had enough motion; treat as failure
                        match_res["status"] = "failed"
                        match_res["reason"] = "no_inlier_in_inner_2_3_or_motion_failed"
                        hist_coords = [0.5, 0.5]
                        prev_coords = [0.5, 0.5]
                        curr_coords = [0.5, 0.5]
                else:
                    # SIFT match failure context (only used in visualization mode)
                    hist_coords = [0.5, 0.5]
                    prev_coords = [0.5, 0.5]
                    curr_coords = [0.5, 0.5]
                    
                sift_match_debug = match_res
                
            # =========================================================
            # Common Sample Processing & Packaging
            # =========================================================
            
            # Generate the 2D attention masks for Channel 1
            hist_mask = generate_attention_mask(
                hist_coords, size=256, mask_type=args.mask_type,
                radius=args.hist_radius, sigma=args.mask_sigma
            )
            prev_mask = generate_attention_mask(
                prev_coords, size=256, mask_type=args.mask_type,
                radius=args.prev_radius, sigma=args.mask_sigma
            )
            zeros_mask = np.zeros((256, 256, 1), dtype=np.float32)
            
            is_success = (sift_match_debug is not None and sift_match_debug.get("status") == "success") or is_hover
            
            # If visualize mode, display HUD preview immediately (including matches and connections)
            if args.visualize:
                dashboard = render_dashboard(
                    f_hist_256, f_prev_256, f_curr_256,
                    hist_mask, prev_mask,
                    hist_coords, prev_coords, curr_coords,
                    sift_info=sift_match_debug,
                    proc_size=args.proc_size,
                    feature_type=args.feature_type
                )
                try:
                    cv2.imshow("Video Dataset Generator Debugger", dashboard)
                    
                    # If auto-running and current attempt is success, stop auto-run and pause
                    if is_success and auto_run_until_success:
                        auto_run_until_success = False
                        
                    # Auto-run mode: short 100ms delay; Manual mode: wait indefinitely (0)
                    delay = 100 if auto_run_until_success else 0
                    key = cv2.waitKey(delay) & 0xFF
                    
                    if key == 13 or key == 10:    # Enter Key
                        auto_run_until_success = True
                    elif key == 32:               # Space Key
                        auto_run_until_success = False
                    elif key == 27 or key == ord('q'):  # Esc or Q to quit
                        cv2.destroyAllWindows()
                        print("\nHUD Preview Mode exited by user.")
                        sys.exit(0)
                except cv2.error as e:
                    print(f"\nGUI Error: Could not render visualization window ({e}).")
                    print("Please run without the '-v' / '--visualize' flag to export pickle files directly.")
                    sys.exit(1)
                    
                # Print status with current attempt count
                if is_success:
                    sample_count += 1
                    print(f"[Sample {sample_count} | Attempt {attempt_count}] Rendered success (Hover: {is_hover})")
                    sys.stdout.flush()
                    attempt_count = 0  # Reset for next sample
                else:
                    print(f"[Sample {sample_count + 1} | Attempt {attempt_count}] Match Failed ({args.feature_type}): {sift_match_debug.get('reason')} - rendering connections.")
                    sys.stdout.flush()
                continue
                
            # Convert to target training formats (Channel 0 = Grayscale, Channel 1 = Mask)
            f_hist_norm = f_hist_256.astype(np.float32) / 255.0
            f_prev_norm = f_prev_256.astype(np.float32) / 255.0
            f_curr_norm = f_curr_256.astype(np.float32) / 255.0
            
            # Stack along last axis (channel axis) to produce shape (256, 256, 2)
            hist_frame = np.stack([f_hist_norm, hist_mask[:, :, 0]], axis=-1)
            prev_frame = np.stack([f_prev_norm, prev_mask[:, :, 0]], axis=-1)
            curr_frame = np.stack([f_curr_norm, zeros_mask[:, :, 0]], axis=-1)
            
            target_heatmap = generate_gaussian_heatmap(curr_coords, size=64, sigma=4.0)
            
            hist_frames_batch.append(hist_frame)
            prev_frames_batch.append(prev_frame)
            curr_frames_batch.append(curr_frame)
            target_heatmaps_batch.append(target_heatmap)
            
            hist_coords_batch.append(hist_coords)
            prev_coords_batch.append(prev_coords)
            curr_coords_batch.append(curr_coords)
            
            sample_count += 1
            attempt_count = 0  # Reset for next sample in non-visual mode
            
        if args.visualize:
            continue
            
        # Serialize batch to pickle matching tracker_model3 expectations exactly
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
            
        print(f"Generated and saved {sample_count}/{total_samples} samples -> {output_file}")
        
    if not args.visualize:
        print("\nVideo dataset generation completed successfully.")

if __name__ == "__main__":
    main()
