#!/usr/bin/env python3
import os
import sys
import argparse
import random
import time
import json
import gc
import hashlib
import pickle
import numpy as np
import cv2

# =====================================================================
# Affine SIFT (ASIFT) and Feature Extraction Helpers
# =====================================================================

def get_affine_tilt_matrix(w, h, t, phi):
    """
    Computes the affine warping matrix for a given tilt t and orientation phi.
    """
    phi_rad = np.deg2rad(phi)
    
    # Rotation matrix to align tilt direction
    R_3x3 = np.array([
        [np.cos(phi_rad), -np.sin(phi_rad), 0.0],
        [np.sin(phi_rad),  np.cos(phi_rad), 0.0],
        [0.0,              0.0,             1.0]
    ], dtype=np.float32)
    
    # Move origin to center before rotation
    T_to_center = np.array([
        [1.0, 0.0, -w / 2.0],
        [0.0, 1.0, -h / 2.0],
        [0.0, 0.0,  1.0]
    ], dtype=np.float32)
    
    T_back = np.array([
        [1.0, 0.0, w / 2.0],
        [0.0, 1.0, h / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # Apply affine tilt (stretch along X-axis by 1/t)
    S_3x3 = np.array([
        [1.0 / t, 0.0, 0.0],
        [0.0,     1.0, 0.0],
        [0.0,     0.0, 1.0]
    ], dtype=np.float32)
    
    # Combined homography-equivalent transform
    M_3x3 = np.dot(T_back, np.dot(R_3x3, np.dot(S_3x3, np.dot(np.linalg.inv(R_3x3), T_to_center))))
    M = M_3x3[:2, :]
    
    # Inverse matrix for projecting coordinates back to the original frame
    M_inv_3x3 = np.linalg.inv(M_3x3)
    M_inv = M_inv_3x3[:2, :]
    
    return M, M_inv

class ASIFTMatcher:
    """
    Affine SIFT implementation using OpenCV SIFT to warp images 
    across 5 highly representative viewpoints.
    """
    def __init__(self):
        self.detector = cv2.SIFT_create()
        self.views = [
            (1.0, 0.0),       # Standard view
            (1.414, 0.0),     # 45 deg tilt along X-axis
            (1.414, 90.0),    # 45 deg tilt along Y-axis
            (2.0, 0.0),       # 60 deg tilt along X-axis
            (2.0, 90.0)       # 60 deg tilt along Y-axis
        ]

    def detect_and_backproject(self, img):
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
                img_warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                
                kps, des = self.detector.detectAndCompute(img_warped, None)
                if des is not None:
                    for kp in kps:
                        pt_w = np.array([kp.pt[0], kp.pt[1], 1.0], dtype=np.float32)
                        pt_orig = np.dot(M_inv, pt_w)
                        
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
# Video Crawling & Image Processing Utility Functions
# =====================================================================

def to_grayscale(frame_raw):
    if frame_raw is None:
        return None
    if len(frame_raw.shape) == 2:
        return frame_raw.copy()
    return cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)

def find_video_files(directory):
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".mpeg", ".mpg", ".webm")
    video_paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(video_extensions):
                video_paths.append(os.path.join(root, f))
    return video_paths

def is_path_in_inner_5_6(path):
    """
    Checks if both hist and curr normalized coordinates are within the inner 5/6 of the frame:
    [1/12, 11/12] which is approximately [0.0833, 0.9167] in normalized coordinates.
    """
    h_x, h_y = path["hist"]
    c_x, c_y = path["curr"]
    
    low = 1.0 / 12.0
    high = 11.0 / 12.0
    
    return (low <= h_x <= high) and (low <= h_y <= high) and (low <= c_x <= high) and (low <= c_y <= high)


# =====================================================================
# Attention Masking and Target Continuous Heatmap Generators
# =====================================================================

def generate_exponential_mask(coords, size=256, sigma=30.0):
    """
    Generates a continuous sharp exponential cone mask: exp(-d / sigma)
    """
    x_target = coords[0] * size
    y_target = coords[1] * size
    
    x = np.arange(0, size, 1, dtype=np.float32)
    y = np.arange(0, size, 1, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)
    
    d = np.sqrt((x_grid - x_target) ** 2 + (y_grid - y_target) ** 2)
    mask = np.exp(-d / sigma)
    
    return np.expand_dims(mask, axis=-1)

def generate_exponential_heatmap(coords, size=64, sigma=4.0):
    """
    Generates a continuous sharp exponential cone target heatmap: exp(-d / sigma)
    """
    x_target = coords[0] * size
    y_target = coords[1] * size
    
    x = np.arange(0, size, 1, dtype=np.float32)
    y = np.arange(0, size, 1, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)
    
    d = np.sqrt((x_grid - x_target) ** 2 + (y_grid - y_target) ** 2)
    heatmap = np.exp(-d / sigma)
    
    return np.expand_dims(heatmap, axis=-1)

# =====================================================================
# Multi-Detector Feature Matching Framework
# =====================================================================

def detect_and_match_single_type(f_hist, f_curr, feature_type, asift_matcher=None, ratio=0.85, min_inliers=6, ransac_thresh=5.0, min_motion=3.0, min_texture_std=3.0, proc_size=800, min_ncc=0.75, keep_debug_info=False):
    """
    Performs keypoint extraction and matching for a single specific detector type.
    Handles L2 vs Hamming metric correctly. Returns verified inlier paths.
    """
    # 1. Detect keypoints & descriptors
    if feature_type == 'asift':
        if asift_matcher is None:
            asift_matcher = ASIFTMatcher()
        kp_hist, des_hist = asift_matcher.detect_and_backproject(f_hist)
        kp_curr, des_curr = asift_matcher.detect_and_backproject(f_curr)
        bf = cv2.BFMatcher(cv2.NORM_L2)
    elif feature_type == 'surf':
        # Safely try loading SURF
        try:
            detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
            kp_hist, des_hist = detector.detectAndCompute(f_hist, None)
            kp_curr, des_curr = detector.detectAndCompute(f_curr, None)
            bf = cv2.BFMatcher(cv2.NORM_L2)
        except Exception:
            # Fallback to standard SIFT if SURF is not available
            detector = cv2.SIFT_create()
            kp_hist, des_hist = detector.detectAndCompute(f_hist, None)
            kp_curr, des_curr = detector.detectAndCompute(f_curr, None)
            bf = cv2.BFMatcher(cv2.NORM_L2)
    elif feature_type == 'akaze':
        detector = cv2.AKAZE_create()
        kp_hist, des_hist = detector.detectAndCompute(f_hist, None)
        kp_curr, des_curr = detector.detectAndCompute(f_curr, None)
        # AKAZE uses binary descriptors (Hamming distance)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:  # SIFT fallback
        detector = cv2.SIFT_create()
        kp_hist, des_hist = detector.detectAndCompute(f_hist, None)
        kp_curr, des_curr = detector.detectAndCompute(f_curr, None)
        bf = cv2.BFMatcher(cv2.NORM_L2)

    if des_hist is None or des_curr is None or len(des_hist) < 2 or len(des_curr) < 2:
        return []

    # 2. Match descriptors using Lowe's Ratio Test
    matches = bf.knnMatch(des_hist, des_curr, k=2)
    valid_matches = []
    for m_list in matches:
        if len(m_list) == 2:
            m, n = m_list
            if m.distance < ratio * n.distance:
                valid_matches.append(m)

    if len(valid_matches) < min_inliers:
        return []

    # Scale spatial thresholds dynamically
    scale_factor = proc_size / 256.0
    scaled_min_motion = min_motion * scale_factor
    r_texture = int(round(5 * scale_factor))
    r_ncc = 10

    pts_hist = []
    pts_curr = []
    kp_triplets = []
    match_distances = []

    # 3. Apply Multi-Stage Geometric & Texture Filters
    for idx_match, m in enumerate(valid_matches):
        pt_h = kp_hist[m.queryIdx].pt
        pt_c = kp_curr[m.trainIdx].pt

        # Filter A: Motion Magnitude (ensure visual shift occurs)
        dx = pt_c[0] - pt_h[0]
        dy = pt_c[1] - pt_h[1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < scaled_min_motion:
            continue

        # Filter B: Local Texture/Variance Check
        hx, hy = int(pt_h[0]), int(pt_h[1])
        if hy - r_texture >= 0 and hy + r_texture + 1 <= proc_size and hx - r_texture >= 0 and hx + r_texture + 1 <= proc_size:
            patch = f_hist[hy-r_texture : hy+r_texture+1, hx-r_texture : hx+r_texture+1]
            if np.std(patch) < min_texture_std:
                continue
        else:
            continue

        # Filter C: Patch Normalized Cross-Correlation (NCC) to reject outliers
        cx, cy = int(pt_c[0]), int(pt_c[1])
        if (hy - r_ncc >= 0 and hy + r_ncc + 1 <= proc_size and hx - r_ncc >= 0 and hx + r_ncc + 1 <= proc_size and
            cy - r_ncc >= 0 and cy + r_ncc + 1 <= proc_size and cx - r_ncc >= 0 and cx + r_ncc + 1 <= proc_size):
            
            patch_h = f_hist[hy-r_ncc : hy+r_ncc+1, hx-r_ncc : hx+r_ncc+1]
            patch_c = f_curr[cy-r_ncc : cy+r_ncc+1, cx-r_ncc : cx+r_ncc+1]
            
            if np.std(patch_h) >= 1e-3 and np.std(patch_c) >= 1e-3:
                res = cv2.matchTemplate(patch_h, patch_c, cv2.TM_CCOEFF_NORMED)
                ncc_val = res[0, 0]
                if ncc_val < min_ncc:
                    continue
            else:
                continue
        else:
            continue

        pts_hist.append(pt_h)
        pts_curr.append(pt_c)
        kp_triplets.append((m.queryIdx, m.trainIdx))
        match_distances.append(m.distance)

    if len(pts_hist) < min_inliers:
        return []

    # 4. RANSAC Epipolar Filtering (Fundamental Matrix)
    pts_hist_np = np.float32(pts_hist)
    pts_curr_np = np.float32(pts_curr)
    
    F, mask = cv2.findFundamentalMat(pts_hist_np, pts_curr_np, cv2.FM_RANSAC, ransac_thresh)
    if F is None or mask is None or F.shape != (3, 3):
        return []

    mask_flat = mask.ravel()
    inlier_paths = []
    inlier_indices = []
    
    for idx, is_inlier in enumerate(mask_flat):
        if is_inlier:
            # Map coordinates back into normalized [0, 1] grid relative to proc_size
            h_norm = [pts_hist[idx][0] / proc_size, pts_hist[idx][1] / proc_size]
            c_norm = [pts_curr[idx][0] / proc_size, pts_curr[idx][1] / proc_size]
            inlier_paths.append({
                "hist": h_norm,
                "curr": c_norm,
                "distance": float(match_distances[idx])
            })
            inlier_indices.append(idx)

    if keep_debug_info:
        # Wrap everything needed to render matches in GUI Mode
        return {
            "status": "success",
            "paths": inlier_paths,
            "kp_hist": kp_hist,
            "kp_curr": kp_curr,
            "triplets": kp_triplets,
            "inliers": inlier_indices
        }
        
    return inlier_paths

def match_features_doublet(f_hist, f_curr, ratio=0.85, min_inliers=6, ransac_thresh=5.0, min_motion=3.0, min_texture_std=3.0, proc_size=800, min_ncc=0.75, asift_matcher=None, keep_debug_info=False):
    """
    Executes and aggregates feature matches across multiple detectors (ASIFT, SURF, AKAZE).
    Keeps each matching metric distinct, and then pools verified moving paths together.
    """
    # If debug info is requested (GUI preview HUD), return detailed matching structures of the first successful type
    if keep_debug_info:
        for f_type in ['asift', 'surf', 'akaze']:
            res = detect_and_match_single_type(
                f_hist, f_curr, f_type, asift_matcher,
                ratio, min_inliers, ransac_thresh, min_motion, min_texture_std, proc_size, min_ncc,
                keep_debug_info=True
            )
            if isinstance(res, dict) and res.get("status") == "success":
                return res
        return {"status": "failed", "paths": [], "kp_hist": [], "kp_curr": [], "triplets": [], "inliers": []}

    aggregated_paths = []
    
    # 1. ASIFT matching
    paths_asift = detect_and_match_single_type(
        f_hist, f_curr, 'asift', asift_matcher,
        ratio, min_inliers, ransac_thresh, min_motion, min_texture_std, proc_size, min_ncc
    )
    if isinstance(paths_asift, list):
        aggregated_paths.extend(paths_asift)

    # 2. SURF matching
    paths_surf = detect_and_match_single_type(
        f_hist, f_curr, 'surf', None,
        ratio, min_inliers, ransac_thresh, min_motion, min_texture_std, proc_size, min_ncc
    )
    if isinstance(paths_surf, list):
        aggregated_paths.extend(paths_surf)

    # 3. AKAZE matching
    paths_akaze = detect_and_match_single_type(
        f_hist, f_curr, 'akaze', None,
        ratio, min_inliers, ransac_thresh, min_motion, min_texture_std, proc_size, min_ncc
    )
    if isinstance(paths_akaze, list):
        aggregated_paths.extend(paths_akaze)

    # Dedup paths
    unique_paths = []
    eps = 1e-4
    for p in aggregated_paths:
        is_duplicate = False
        for up in unique_paths:
            dist_h = np.sqrt((p["hist"][0] - up["hist"][0])**2 + (p["hist"][1] - up["hist"][1])**2)
            dist_c = np.sqrt((p["curr"][0] - up["curr"][0])**2 + (p["curr"][1] - up["curr"][1])**2)
            if dist_h < eps and dist_c < eps:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_paths.append(p)

    return unique_paths

# =====================================================================
# Hover / Jitter Simulation & Helper Drawing Blocks
# =====================================================================

def simulate_hover_jitter(frame_256, coords):
    """
    Simulates drone hovering and local camera jitter (gimbal vibrations, wind)
    by applying small random rotation and translation matrices to a static frame.
    """
    theta = np.random.uniform(-3.0, 3.0)  # Random rotation +/- 3 degrees
    dx = np.random.uniform(-5.0, 5.0)    # Random translation +/- 5 pixels
    dy = np.random.uniform(-5.0, 5.0)
    
    M = cv2.getRotationMatrix2D((128.0, 128.0), theta, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    
    warped_frame = cv2.warpAffine(frame_256, M, (256, 256), borderMode=cv2.BORDER_REPLICATE)
    
    x_px = coords[0] * 256.0
    y_px = coords[1] * 256.0
    pt = np.array([x_px, y_px, 1.0], dtype=np.float32)
    warped_pt = np.dot(M, pt)
    
    warped_norm = [
        np.clip(warped_pt[0] / 256.0, 0.0, 1.0),
        np.clip(warped_pt[1] / 256.0, 0.0, 1.0)
    ]
    return warped_frame, warped_norm

def draw_hud_label(img, label, org, color=(0, 255, 0), scale=0.38, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(label, font, scale, thickness)
    
    y1 = max(org[1] - h - 4, 0)
    y2 = min(org[1] + baseline, img.shape[0])
    x1 = max(org[0] - 4, 0)
    x2 = min(org[0] + w + 4, img.shape[1])
    
    sub_img = img[y1:y2, x1:x2]
    if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
        rect = np.zeros_like(sub_img)
        rect[:] = 15
        cv2.addWeighted(sub_img, 0.3, rect, 0.7, 0, sub_img)
    
    cv2.putText(img, label, org, font, scale, color, thickness, cv2.LINE_AA)

def draw_epipolar_line(img, line, color, thickness=1):
    a, b, c = line[0], line[1], line[2]
    h, w = img.shape[:2]
    if abs(b) > abs(a):
        x0 = 0
        y0 = int(round(-c / b))
        x1 = w
        y1 = int(round(-(c + a * w) / b))
    else:
        y0 = 0
        x0 = int(round(-c / a))
        y1 = h
        x1 = int(round(-(c + b * h) / a))
    cv2.line(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)

# =====================================================================
# Dashboard Rendering for Symmetrical 2-Frame Pipeline
# =====================================================================

def render_dashboard(f_hist_256, f_curr_256, hist_mask, hist_norm, curr_norm, sift_info=None, proc_size=800, label_radius=32, match_index=0, total_matches=0, match_dist=0.0):
    """
    Renders a dynamic, resolution-generic 2-row HUD dashboard tailored strictly for Two-Frame Siamese Tracking.
    All dimensions (panels, font scales, text positions) are computed dynamically in real time based on proc_size.
    """
    # 1. Dynamic Screen Resolution Detection (cached on the function to prevent Tkinter window rebuild overhead)
    if not hasattr(render_dashboard, "W"):
        try:
            import tkinter as tk
            root = tk.Tk()
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            root.destroy()
        except Exception:
            # Fallback to standard 1080p if GUI environment is not fully ready
            screen_w = 1920
            screen_h = 1080

        # We want the window width to comfortably fit in 80% of screen width,
        # and the height (35 + W//4 + W//9) to comfortably fit in 80% of screen height.
        limit_w_by_width = int(screen_w * 0.80)
        limit_w_by_height = int((screen_h * 0.80 - 35) * 36 / 13)
        
        target_W = min(limit_w_by_width, limit_w_by_height)
        W = int(round(target_W / 36.0)) * 36
        W = max(W, 720) # Clamp minimum width to 720px
        
        render_dashboard.W = W
        
    W = render_dashboard.W
    
    # Calculate high-res (Row 1) and compact (Row 2) panel sizes
    h_size = W // 4
    s_size = W // 9
    
    # Proportional scaling factors for text and drawing elements
    f_scale = h_size / 288.0
    font_scale_h = 0.38 * f_scale
    title_font_scale_h = 0.45 * f_scale
    thickness_h = max(1, int(round(1.0 * f_scale)))
    
    # 2. Resize and prepare main Row 1 context images (h_size x h_size)
    f_hist_h = cv2.resize(f_hist_256, (h_size, h_size), interpolation=cv2.INTER_LINEAR)
    f_curr_h = cv2.resize(f_curr_256, (h_size, h_size), interpolation=cv2.INTER_LINEAR)
    
    h_color = cv2.cvtColor(f_hist_h, cv2.COLOR_GRAY2BGR)
    c_color = cv2.cvtColor(f_curr_h, cv2.COLOR_GRAY2BGR)
    
    # Soft red glow attention mask scaled to h_size x h_size
    h_mask_h = cv2.resize(hist_mask, (h_size, h_size), interpolation=cv2.INTER_LINEAR)
    h_mask_bgr = np.zeros_like(h_color)
    h_mask_bgr[:, :, 2] = (h_mask_h * 255.0).astype(np.uint8)
    cv2.addWeighted(h_mask_bgr, 0.50, h_color, 1.0, 0, h_color)
    
    is_success = (sift_info is not None and sift_info.get("status") == "success")
    if is_success:
        status_color = (0, 255, 0)
    elif sift_info is None:
        status_color = (0, 255, 255)
    else:
        status_color = (0, 0, 255)
        
    sift_panel = np.hstack([f_hist_h, f_curr_h])
    s_color = cv2.cvtColor(sift_panel, cv2.COLOR_GRAY2BGR)
    
    inlier_count = 0
    avg_err = 0.0
    
    # Flow Vector Panel (starts as copy of f_hist_h)
    flow_panel = cv2.cvtColor(f_hist_h, cv2.COLOR_GRAY2BGR)
    
    sift_label = "GEOMETRIC CONNECTIONS (HIST -> CURR)" if is_success else "GEOMETRIC FAILURE CONNECTIONS"
    sift_label_color = (255, 255, 0) if is_success else (0, 128, 255)
    
    # 3. Draw matching keypoint targets on main panels dynamically scaled to h_size
    if sift_info is not None and "paths" in sift_info and len(sift_info["paths"]) > 0:
        inliers = sift_info["inliers"]
        inlier_count = len(inliers)
        
        for p in sift_info["paths"]:
            px_h = int(p["hist"][0] * h_size)
            py_h = int(p["hist"][1] * h_size)
            px_c = int(p["curr"][0] * h_size)
            py_c = int(p["curr"][1] * h_size)
            
            p1_h = (px_h, py_h)
            p2_c = (px_c + h_size, py_c)
            p2_flow = (px_c, py_c)
            
            is_active = (abs(p["hist"][0] - hist_norm[0]) < 1e-5 and abs(p["hist"][1] - hist_norm[1]) < 1e-5)
            
            if is_active:
                # Active target: draw larger indicators on Row 1 context images
                cv2.circle(h_color, p1_h, int(round(5 * f_scale)), (0, 0, 255), -1)
                cv2.circle(c_color, p2_flow, int(round(8 * f_scale)), (0, 255, 0), max(1, int(round(2 * f_scale))))
                cv2.circle(c_color, p2_flow, int(round(2 * f_scale)), (0, 255, 0), -1)
                
                # Active match: thick blue connection on Row 1 Match panel
                cv2.line(s_color, p1_h, p2_c, (255, 0, 0), max(1, int(round(2 * f_scale))), cv2.LINE_AA)
                cv2.circle(s_color, p1_h, int(round(5 * f_scale)), (255, 0, 0), -1)
                cv2.circle(s_color, p2_c, int(round(5 * f_scale)), (255, 0, 0), -1)
                
                # Active flow vector: thick yellow vector on Row 1 Flow panel
                cv2.line(flow_panel, p1_h, p2_flow, (0, 255, 255), max(1, int(round(2 * f_scale))), cv2.LINE_AA)
                cv2.circle(flow_panel, p1_h, int(round(4 * f_scale)), (0, 0, 255), -1)
            else:
                # Background matches: draw smaller indicators on Row 1 context images
                cv2.circle(h_color, p1_h, int(round(3 * f_scale)), (0, 0, 180), -1)
                cv2.circle(c_color, p2_flow, int(round(5 * f_scale)), (0, 180, 0), 1)
                cv2.circle(c_color, p2_flow, int(round(1 * f_scale)), (0, 180, 0), -1)
                
                # Background matches: thin white connections on Row 1 Match panel
                cv2.line(s_color, p1_h, p2_c, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.circle(s_color, p1_h, int(round(3 * f_scale)), (220, 220, 220), -1)
                cv2.circle(s_color, p2_c, int(round(3 * f_scale)), (220, 220, 220), -1)
                
                # Background flow vectors: thin green vectors on Row 1 Flow panel
                cv2.line(flow_panel, p1_h, p2_flow, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.circle(flow_panel, p1_h, int(round(2 * f_scale)), (0, 0, 255), -1)
    else:
        hx, hy = int(hist_norm[0] * h_size), int(hist_norm[1] * h_size)
        cv2.circle(h_color, (hx, hy), int(round(4 * f_scale)), (0, 0, 255), -1)
        
        cx, cy = int(curr_norm[0] * h_size), int(curr_norm[1] * h_size)
        cv2.circle(c_color, (cx, cy), int(round(8 * f_scale)), (0, 255, 0), max(1, int(round(2 * f_scale))))
        cv2.circle(c_color, (cx, cy), int(round(2 * f_scale)), (0, 255, 0), -1)
        
        text_scale = 0.6 * f_scale
        text_thickness = max(1, int(round(1.5 * f_scale)))
        
        if sift_info is None:
            cv2.putText(s_color, "HOVER MODE", (int(160 * f_scale), int(145 * f_scale)), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), text_thickness, cv2.LINE_AA)
            cv2.putText(s_color, "Gimbal Jitter Active", (int(180 * f_scale), int(175 * f_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * f_scale, (0, 200, 200), thickness_h, cv2.LINE_AA)
        else:
            cv2.putText(s_color, "NO MATCHES FOUND", (int(140 * f_scale), int(145 * f_scale)), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)
            
    # 4. Create high-resolution HUD System Stats console (Row 1, Column 4)
    system_stats_panel = np.zeros((h_size, h_size, 3), dtype=np.uint8) + 15
    cv2.rectangle(system_stats_panel, (0, 0), (h_size - 1, h_size - 1), status_color, max(1, int(round(2 * f_scale))))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (220, 220, 220)
    label_color = (0, 255, 255)
    
    cv2.putText(system_stats_panel, "--- SYSTEM STATUS ---", (int(70 * f_scale), int(35 * f_scale)), font, title_font_scale_h, status_color, thickness_h, cv2.LINE_AA)
    cv2.putText(system_stats_panel, "Siam-Lite 2-Frame Tracker", (int(20 * f_scale), int(75 * f_scale)), font, font_scale_h, text_color, thickness_h, cv2.LINE_AA)
    cv2.putText(system_stats_panel, f"Active Matches: {inlier_count}", (int(20 * f_scale), int(115 * f_scale)), font, font_scale_h, label_color, thickness_h, cv2.LINE_AA)
    cv2.putText(system_stats_panel, "Focus of Expansion: Active", (int(20 * f_scale), int(155 * f_scale)), font, font_scale_h, text_color, thickness_h, cv2.LINE_AA)
    cv2.putText(system_stats_panel, "Boundary Limits: 5/6", (int(20 * f_scale), int(195 * f_scale)), font, font_scale_h, text_color, thickness_h, cv2.LINE_AA)
    cv2.putText(system_stats_panel, f"Proc Size: {proc_size} px", (int(20 * f_scale), int(235 * f_scale)), font, font_scale_h, text_color, thickness_h, cv2.LINE_AA)
    cv2.putText(system_stats_panel, "Mode: Sourcing-Safe HUD", (int(20 * f_scale), int(275 * f_scale)), font, font_scale_h, text_color, thickness_h, cv2.LINE_AA)
    
    # 5. Construct Row 2 secondary panels (s_size x s_size each)
    f_curr_s = cv2.resize(f_curr_256, (s_size, s_size), interpolation=cv2.INTER_AREA)
    epipoles_on_curr = cv2.cvtColor(f_curr_s, cv2.COLOR_GRAY2BGR)
    
    if sift_info is not None and len(sift_info.get("inliers", [])) >= 5:
        kp_hist = sift_info["kp_hist"]
        kp_curr = sift_info["kp_curr"]
        triplets = sift_info["triplets"]
        inliers = sift_info["inliers"]
        scale_s = float(s_size) / proc_size
        
        pts_h_s = []
        pts_c_s = []
        
        for idx in inliers:
            if idx < len(triplets):
                trip = triplets[idx]
                pt_h = kp_hist[trip[0]].pt
                pt_c = kp_curr[trip[1]].pt
                pts_h_s.append([pt_h[0] * scale_s, pt_h[1] * scale_s])
                pts_c_s.append([pt_c[0] * scale_s, pt_c[1] * scale_s])
                
        pts_h_s = np.float32(pts_h_s)
        pts_c_s = np.float32(pts_c_s)
        
        if len(pts_h_s) >= 8:
            F_s, _ = cv2.findFundamentalMat(pts_h_s, pts_c_s, cv2.FM_RANSAC, 5.0 * (float(s_size) / 256.0))
        else:
            F_s, _ = cv2.findFundamentalMat(pts_h_s, pts_c_s, cv2.FM_LMEDS)
            
        if F_s is not None and F_s.shape == (3, 3):
            lines_on_curr = cv2.computeCorrespondEpilines(pts_h_s.reshape(-1, 1, 2), 1, F_s)
            
            total_err = 0.0
            err_count = 0
            
            for i in range(len(pts_h_s)):
                pt_c = pts_c_s[i]
                l_curr = lines_on_curr[i][0]
                
                a, b, c = l_curr[0], l_curr[1], l_curr[2]
                denom = np.sqrt(a**2 + b**2)
                if denom > 1e-8:
                    dist = abs(a * pt_c[0] + b * pt_c[1] + c) / denom
                    total_err += dist * (proc_size / float(s_size))
                    err_count += 1
                    
                color = (255, 255, 255) if i > 0 else (255, 0, 0)
                thickness = 1
                radius = 2 if i > 0 else 3
                
                draw_epipolar_line(epipoles_on_curr, l_curr, color, thickness)
                cv2.circle(epipoles_on_curr, (int(pt_c[0]), int(pt_c[1])), radius, color, -1)
                
            if err_count > 0:
                avg_err = total_err / err_count
                
    cv2.rectangle(epipoles_on_curr, (0, 0), (s_size - 1, s_size - 1), (255, 0, 255) if is_success else (0, 0, 255), 1)
    
    # 6. Row 2 secondary telemetry and mask panels (scaled to s_size)
    telemetry_scale = s_size / 128.0
    telemetry_font_scale = 0.30 * telemetry_scale
    telemetry_title_font_scale = 0.35 * telemetry_scale
    telemetry_thickness = max(1, int(round(1.0 * telemetry_scale)))
    
    telemetry_panel = np.zeros((s_size, s_size, 3), dtype=np.uint8) + 20
    cv2.rectangle(telemetry_panel, (0, 0), (s_size - 1, s_size - 1), status_color, 1)
    cv2.putText(telemetry_panel, "TELEMETRY", (int(25 * telemetry_scale), int(20 * telemetry_scale)), font, telemetry_title_font_scale, status_color, telemetry_thickness, cv2.LINE_AA)
    cv2.putText(telemetry_panel, f"Inliers: {inlier_count}", (int(10 * telemetry_scale), int(50 * telemetry_scale)), font, telemetry_font_scale, text_color, telemetry_thickness, cv2.LINE_AA)
    if total_matches > 0:
        cv2.putText(telemetry_panel, f"Rank: {match_index+1}/{total_matches}", (int(10 * telemetry_scale), int(80 * telemetry_scale)), font, telemetry_font_scale, text_color, telemetry_thickness, cv2.LINE_AA)
        cv2.putText(telemetry_panel, f"Dist: {match_dist:.3f}", (int(10 * telemetry_scale), int(110 * telemetry_scale)), font, telemetry_font_scale, label_color, telemetry_thickness, cv2.LINE_AA)
    else:
        avg_err_str = f"EpiErr: {avg_err:.2f}" if avg_err > 0.0 else "EpiErr: N/A"
        cv2.putText(telemetry_panel, avg_err_str, (int(10 * telemetry_scale), int(80 * telemetry_scale)), font, telemetry_font_scale, text_color, telemetry_thickness, cv2.LINE_AA)
        
    # Mask Panel
    hist_mask_s = cv2.resize(hist_mask, (s_size, s_size), interpolation=cv2.INTER_LINEAR)
    hist_mask_panel = np.zeros((s_size, s_size, 3), dtype=np.uint8)
    hist_mask_panel[:, :, 2] = (hist_mask_s * 255.0).astype(np.uint8)
    cv2.rectangle(hist_mask_panel, (0, 0), (s_size - 1, s_size - 1), (0, 0, 255), 1)
    cv2.putText(hist_mask_panel, "MASK", (int(10 * telemetry_scale), int(115 * telemetry_scale)), font, telemetry_font_scale, (0, 0, 255), telemetry_thickness, cv2.LINE_AA)
    
    # Heatmap Panel
    expected_heatmap_64 = generate_exponential_heatmap(curr_norm, size=64, sigma=label_radius / 8.0)
    expected_heatmap_s = cv2.resize(expected_heatmap_64, (s_size, s_size), interpolation=cv2.INTER_LINEAR)
    expected_heatmap_panel = np.zeros((s_size, s_size, 3), dtype=np.uint8)
    expected_heatmap_panel[:, :, 1] = (expected_heatmap_s * 255.0).astype(np.uint8)
    cv2.rectangle(expected_heatmap_panel, (0, 0), (s_size - 1, s_size - 1), (0, 255, 0), 1)
    cv2.putText(expected_heatmap_panel, "HEATMAP", (int(10 * telemetry_scale), int(115 * telemetry_scale)), font, telemetry_font_scale, (0, 255, 0), telemetry_thickness, cv2.LINE_AA)
    
    # 7. Draw HUD labels dynamically scaled on main high-res panels
    draw_hud_label(s_color, sift_label, (int(10 * f_scale), int((h_size - 19) * f_scale)), sift_label_color, scale=font_scale_h, thickness=thickness_h)
    draw_hud_label(flow_panel, "FLOW VECTORS: HIST TO CURR", (int(10 * f_scale), int((h_size - 19) * f_scale)), (0, 255, 255) if is_success else (0, 0, 255), scale=font_scale_h, thickness=thickness_h)
    
    # 8. Spacer panels for Row 2 (s_size x s_size each)
    spacer_s = np.zeros((s_size, s_size, 3), dtype=np.uint8) + 15
    cv2.rectangle(spacer_s, (0, 0), (s_size - 1, s_size - 1), (35, 35, 35), 1)
    
    # 9. Horizontal stacking of rows
    row1 = np.hstack([s_color, flow_panel, system_stats_panel])
    row2 = np.hstack([
        epipoles_on_curr, telemetry_panel, hist_mask_panel, expected_heatmap_panel,
        spacer_s, spacer_s, spacer_s, spacer_s, spacer_s
    ])
    
    dashboard = np.vstack([row1, row2])
    
    # Header bar with dynamically right-aligned help commands
    header_bar = np.zeros((35, W, 3), dtype=np.uint8)
    title_text = "VIDEO DATASET GENERATOR - PREVIEW HUD (2-FRAME LITE)"
    title_color = (0, 255, 255) if (is_success or sift_info is None) else (0, 0, 255)
    
    cv2.putText(header_bar, title_text, (10, 22), font, 0.40, title_color, 1, cv2.LINE_AA)
    
    help_text = "[SPACE]: Next | [ENTER]: Auto-run | [ESC/Q]: Exit"
    (help_w, _), _ = cv2.getTextSize(help_text, font, 0.30, 1)
    cv2.putText(header_bar, help_text, (W - help_w - 15, 22), font, 0.30, (200, 200, 200), 1, cv2.LINE_AA)
    
    final_output = np.vstack([header_bar, dashboard])
    return final_output
# Real-Time Visualization Preview Pipeline (ASIFT + SURF + AKAZE)
# =====================================================================

def visualize_pipeline(args, video_paths, asift_matcher):
    print("\n=== ENTERING HUD PREVIEW MODE (2-FRAME LITE) ===")
    print("Rendering generated sequences in real-time. No files will be exported to disk.")
    print("Controls: Press [SPACE] for next single step, [ENTER] to auto-run until success, [ESC/Q] to exit.\n")
    sys.stdout.flush()
    
    try:
        cv2.namedWindow("Video Dataset Generator Debugger", cv2.WINDOW_AUTOSIZE)
    except cv2.error as e:
        print(f"Warning: Could not initialize visual window ({e}).")
        print("If you are running in a headless environment, please run without '-v' / '--visualize'.")
        sys.exit(1)
        
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
        
        if not cap.isOpened():
            print(f"[ERROR] Corrupted video file: {os.path.abspath(random_video)}")
            sys.stdout.flush()
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 200:
            print(f"[ERROR] Too few frames: {os.path.abspath(random_video)}")
            sys.stdout.flush()
            cap.release()
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
            
        gap_frames = int(round(args.frame_gap_seconds * fps))
        max_start = total_frames - int(fps * 1.5)
        start_frame_idx = random.randint(0, max_start)
        
        # Read hist frame
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
            f_curr_256, curr_coords = simulate_hover_jitter(f_hist_256, target_coords)
            hist_coords = target_coords
            sift_match_debug = None
        else:
            # Read curr frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx + gap_frames)
            ret_c, frame_curr_raw = cap.read()
            cap.release()
            if not ret_c or frame_curr_raw is None:
                continue
                
            f_curr_full = to_grayscale(frame_curr_raw)
            f_curr_256 = cv2.resize(f_curr_full, (256, 256), interpolation=cv2.INTER_AREA)
            
            f_hist_proc = cv2.resize(f_hist_full, (args.proc_size, args.proc_size), interpolation=cv2.INTER_AREA)
            f_curr_proc = cv2.resize(f_curr_full, (args.proc_size, args.proc_size), interpolation=cv2.INTER_AREA)
            
            # Match doublets and retain debug structures for GUI
            match_res = match_features_doublet(
                f_hist_proc, f_curr_proc,
                ratio=args.ratio, min_inliers=args.min_inliers,
                ransac_thresh=args.ransac_thresh,
                min_motion=args.target_min_motion,
                min_texture_std=args.min_texture_std,
                proc_size=args.proc_size,
                min_ncc=args.min_ncc,
                asift_matcher=asift_matcher,
                keep_debug_info=True
            )
            
            if isinstance(match_res, dict) and match_res.get("status") == "success":
                valid_paths = [p for p in match_res["paths"] if is_path_in_inner_5_6(p)]
                valid_paths.sort(key=lambda x: x.get("distance", float('inf')))
                
                if len(valid_paths) > 0:
                    selected_path = valid_paths[0]
                    hist_coords = selected_path["hist"]
                    curr_coords = selected_path["curr"]
                    match_dist = selected_path.get("distance", 0.0)
                    is_success = True
                    total_matches = len(valid_paths)
                else:
                    match_res["status"] = "failed"
                    match_res["reason"] = "no_inlier_in_inner_5_6"
                    hist_coords = [0.5, 0.5]
                    curr_coords = [0.5, 0.5]
                    is_success = False
                    total_matches = 0
                    match_dist = 0.0
            else:
                hist_coords = [0.5, 0.5]
                curr_coords = [0.5, 0.5]
                is_success = False
                total_matches = 0
                match_dist = 0.0
                
            sift_match_debug = match_res
            
        # Draw and display the HUD panel
        if is_hover:
            hist_coords = target_coords
            curr_coords = curr_coords
            is_success = True
            total_matches = 0
            match_dist = 0.0
            
        hist_mask = generate_exponential_mask(
            hist_coords, size=256, sigma=args.mask_sigma
        )
        
        dashboard = render_dashboard(
            f_hist_256, f_curr_256,
            hist_mask, hist_coords, curr_coords,
            sift_info=sift_match_debug,
            proc_size=args.proc_size,
            label_radius=args.label_radius,
            match_index=0,
            total_matches=total_matches,
            match_dist=match_dist
        )
        
        try:
            cv2.imshow("Video Dataset Generator Debugger", dashboard)
            if is_success and auto_run_until_success:
                auto_run_until_success = False
            delay = 100 if auto_run_until_success else 0
            key = cv2.waitKey(delay) & 0xFF
            
            if key in [13, 10]:  # ENTER
                auto_run_until_success = True
            elif key == 32:  # SPACE
                auto_run_until_success = False
            elif key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q
                cv2.destroyAllWindows()
                print("\nHUD Preview Mode exited by user.")
                sys.exit(0)
        except cv2.error as e:
            print(f"\nGUI Error: Could not render visualization window ({e}).")
            sys.exit(1)

        # Logging output for this step
        if is_success:
            sample_count += 1
            print(f"[Sample {sample_count} | Attempt {attempt_count}] success (Hover: {is_hover})")
            sys.stdout.flush()
            attempt_count = 0
        else:
            print(f"[Sample {sample_count + 1} | Attempt {attempt_count}] Match Failed: {sift_match_debug.get('reason')} - rendering connections.")
            sys.stdout.flush()

# =====================================================================
# Production Two-Stage High-Throughput Pipeline
# =====================================================================

def run_two_stage_pipeline(args, video_paths, asift_matcher):
    print("\n===========================================================")
    print("=== TargetTracker3-Lite High-Throughput Dataset Pipe ===")
    print("===========================================================\n")
    sys.stdout.flush()
    
    tmp_dir = os.path.join(args.output_dir, "tmp_raw_extracted")
    os.makedirs(tmp_dir, exist_ok=True)
    
    video_lengths = {}
    video_fps_map = {}
    
    print("[*] Pre-mapping video durations...")
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
    print(f"[+] Mapped {len(video_lengths)} active videos.\n")
    sys.stdout.flush()
    
    # Filter completed videos
    processed_files = os.listdir(tmp_dir)
    processed_hashes = set()
    for filename in processed_files:
        if filename.startswith("processed_") and filename.endswith(".json"):
            parts = filename[:-5].split("_")
            if len(parts) >= 3:
                processed_hashes.add(parts[-1])
                
    remaining_videos = [p for p in video_paths if hashlib.md5(p.encode()).hexdigest()[:8] not in processed_hashes]
    
    print(f"[*] Resume Status: {len(video_paths) - len(remaining_videos)}/{len(video_paths)} videos already completed.")
    print(f"[*] Remaining videos to process: {len(remaining_videos)}\n")
    sys.stdout.flush()
    
    for idx, path in enumerate(remaining_videos):
        video_name = os.path.basename(path)
        v_hash = hashlib.md5(path.encode()).hexdigest()[:8]
        metadata_filename = f"processed_{os.path.splitext(video_name)[0]}_{v_hash}.json"
        metadata_path = os.path.join(tmp_dir, metadata_filename)
        
        print(f"\n[STAGING STAGE 1] Video {idx+1}/{len(remaining_videos)}: {video_name}")
        sys.stdout.flush()
        
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[ERROR] Corrupted video: {os.path.abspath(path)}")
            with open(metadata_path, "w") as f:
                json.dump({"video_path": path, "sample_count": 0, "status": "corrupted"}, f)
            continue
            
        total_frames = video_lengths.get(path, 200)
        fps = video_fps_map.get(path, 30.0)
        
        if total_frames < 100:
            print(f"[ERROR] Too few frames: {os.path.abspath(path)}")
            cap.release()
            with open(metadata_path, "w") as f:
                json.dump({"video_path": path, "sample_count": 0, "status": "too_few_frames"}, f)
            continue
            
        frame_step = int(round(args.temporal_step_seconds * fps))
        gap_frames = int(round(args.frame_gap_seconds * fps))
        
        video_sample_count = 0
        consecutive_failures = 0
        current_frame_idx = 0
        max_start = total_frames - int(fps * 1.5)
        
        t_start_video = time.time()
        doublets_processed = 0
        successful_frames = 0
        
        while current_frame_idx <= max_start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame_hist_raw = cap.read()
            if not ret or frame_hist_raw is None:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx + gap_frames)
            ret_c, frame_curr_raw = cap.read()
            if not ret_c or frame_curr_raw is None:
                break
                
            doublets_processed += 1
            
            f_hist_full = to_grayscale(frame_hist_raw)
            f_hist_256 = cv2.resize(f_hist_full, (256, 256), interpolation=cv2.INTER_AREA)
            f_hist_proc = cv2.resize(f_hist_full, (args.proc_size, args.proc_size), interpolation=cv2.INTER_AREA)
            
            f_curr_full = to_grayscale(frame_curr_raw)
            f_curr_256 = cv2.resize(f_curr_full, (256, 256), interpolation=cv2.INTER_AREA)
            f_curr_proc = cv2.resize(f_curr_full, (args.proc_size, args.proc_size), interpolation=cv2.INTER_AREA)
            
            inlier_paths = match_features_doublet(
                f_hist_proc, f_curr_proc,
                ratio=args.ratio, min_inliers=args.min_inliers,
                ransac_thresh=args.ransac_thresh,
                min_motion=args.target_min_motion,
                min_texture_std=args.min_texture_std,
                proc_size=args.proc_size,
                min_ncc=args.min_ncc,
                asift_matcher=asift_matcher
            )
            
            frame_yield = 0
            if len(inlier_paths) > 0:
                consecutive_failures = 0
                
                for path_candidate in inlier_paths:
                    if is_path_in_inner_5_6(path_candidate):
                        h_pt = path_candidate["hist"]
                        c_pt = path_candidate["curr"]
                        
                        hist_mask = generate_exponential_mask(
                            h_pt, size=256, sigma=args.mask_sigma
                        )
                        
                        f_hist_norm = f_hist_256.astype(np.float32) / 255.0
                        f_curr_norm = f_curr_256.astype(np.float32) / 255.0
                        
                        hist_frame = np.stack([f_hist_norm, hist_mask[:, :, 0]], axis=-1)
                        curr_frame = np.expand_dims(f_curr_norm, axis=-1)
                        
                        target_heatmap = generate_exponential_heatmap(c_pt, size=64, sigma=args.label_radius / 8.0)
                        
                        sample_data = {
                            "hist_frame": hist_frame,
                            "curr_frame": curr_frame,
                            "target_heatmap": target_heatmap,
                            "debug_coords": {
                                "hist": h_pt,
                                "curr": c_pt
                            }
                        }
                        
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
                
            current_frame_idx += frame_step
            
            elapsed = time.time() - t_start_video
            rate = doublets_processed / elapsed if elapsed > 0 else 0.0
            
            remaining_doublets = max(0, (max_start - current_frame_idx) // frame_step)
            eta_current_sec = remaining_doublets / rate if rate > 0 else 0.0
            
            percent_complete = int((current_frame_idx / max_start) * 100) if max_start > 0 else 0
            percent_complete = min(100, max(0, percent_complete))
            
            current_time_str = time.strftime('%M:%S', time.gmtime(int(current_frame_idx / fps)))
            total_time_str = time.strftime('%M:%S', time.gmtime(int(total_frames / fps)))
            eta_curr_str = f"{int(eta_current_sec // 60)}m {int(eta_current_sec % 60)}s" if eta_current_sec > 0 else "N/A"
            
            sys.stdout.write(
                f"\r[STAGING STAGE 1] Progress: {percent_complete}% ({current_time_str}/{total_time_str}) | "
                f"Yield: {successful_frames} frames ({video_sample_count} samples) | "
                f"Rate: {rate:.1f} Hz | ETA Video: {eta_curr_str}   "
            )
            sys.stdout.flush()
            
            del f_hist_full, f_hist_256, f_hist_proc
            del f_curr_full, f_curr_256, f_curr_proc
            del frame_hist_raw, frame_curr_raw
            del inlier_paths
            gc.collect()
            
            if consecutive_failures >= args.max_consecutive_failures:
                print(f"\n[WARNING] Consecutive failures reached ({consecutive_failures}). Skipping video.")
                break
                
        cap.release()
        
        with open(metadata_path, "w") as f:
            json.dump({
                "video_path": path,
                "sample_count": video_sample_count,
                "status": "success",
                "v_hash": v_hash
            }, f)
            
        print(f"\n[SUCCESS] Completed {video_name}. Yielded {video_sample_count} samples.")
        sys.stdout.flush()
        gc.collect()
        
    # STAGE 2
    print("\n===========================================================")
    print("=== STAGE 2: SHUFFLED I.I.D. TRAINING DATASET COMPILATION ===")
    print("===========================================================\n")
    sys.stdout.flush()
    
    sample_files = [f for f in os.listdir(tmp_dir) if f.startswith("sample_") and f.endswith(".pkl")]
    total_samples = len(sample_files)
    
    print(f"[*] Total compiled samples extracted: {total_samples}")
    if total_samples == 0:
        print("[ERROR] No valid dataset samples were extracted in Stage 1.")
        sys.exit(1)
        
    print("[*] Shuffling sample filenames globally to guarantee i.i.d. balance...")
    random.shuffle(sample_files)
    print("[+] Shuffling complete.\n")
    sys.stdout.flush()
    
    num_batches = int(np.ceil(total_samples / args.batch_size))
    print(f"[*] Packaging {total_samples} samples into {num_batches} batches of size {args.batch_size}...")
    sys.stdout.flush()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for b in range(num_batches):
        batch_slice = sample_files[b * args.batch_size : (b + 1) * args.batch_size]
        
        if len(batch_slice) < args.batch_size:
            pad_needed = args.batch_size - len(batch_slice)
            for i in range(pad_needed):
                batch_slice.append(sample_files[i % len(sample_files)])
                
        hist_frames_batch = []
        curr_frames_batch = []
        target_heatmaps_batch = []
        
        hist_coords_batch = []
        curr_coords_batch = []
        
        for sample_filename in batch_slice:
            sample_path = os.path.join(tmp_dir, sample_filename)
            try:
                with open(sample_path, "rb") as f_in:
                    sample = pickle.load(f_in)
                    
                hist_frames_batch.append(sample["hist_frame"])
                curr_frames_batch.append(sample["curr_frame"])
                target_heatmaps_batch.append(sample["target_heatmap"])
                
                hist_coords_batch.append(sample["debug_coords"]["hist"])
                curr_coords_batch.append(sample["debug_coords"]["curr"])
            except Exception as e:
                print(f"[ERROR] Failed to load sample file '{sample_path}': {e}")
                sys.exit(1)
                
        batch_data = {
            "inputs": [
                np.array(hist_frames_batch, dtype=np.float32),
                np.array(curr_frames_batch, dtype=np.float32)
            ],
            "targets": np.array(target_heatmaps_batch, dtype=np.float32),
            "debug_coords": {
                "hist": np.array(hist_coords_batch, dtype=np.float32),
                "curr": np.array(curr_coords_batch, dtype=np.float32)
            }
        }
        
        output_file = os.path.join(args.output_dir, f"video_dataset_{b}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(batch_data, f)
            
        print(f"Generated final batch {b+1}/{num_batches} -> {output_file}")
        sys.stdout.flush()
        
        del batch_data, hist_frames_batch, curr_frames_batch, target_heatmaps_batch
        del hist_coords_batch, curr_coords_batch
        gc.collect()
        
    print("\n===========================================================")
    print("=== HIGH-THROUGHPUT TWO-STAGE DATASET GENERATION COMPLETE ===")
    print("===========================================================\n")
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(
        description="Generates continuous symmetric 2-frame datasets using ASIFT, SURF, and AKAZE."
    )
    parser.add_argument(
        "videos_dir",
        help="Path to the directory containing video files."
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="video_dataset",
        help="Output directory path (default: video_dataset)."
    )
    parser.add_argument(
        "-v", "--visualize",
        action="store_true",
        help="Enables interactive preview mode rendering SIFT matches in real-time."
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
        help="Lowe's ratio threshold for match filtering (default: 0.85)."
    )
    parser.add_argument(
        "--min_inliers",
        type=int,
        default=6,
        help="Minimum verified RANSAC inliers to accept a pair (default: 6)."
    )
    parser.add_argument(
        "--hover_prob",
        type=float,
        default=0.05,
        help="Probability of selecting stationary hovering with gimbal camera shake (default: 0.05)."
    )
    parser.add_argument(
        "--mask_sigma",
        type=float,
        default=30.0,
        help="Sigma of the historical exponential mask in pixels (default: 30.0)."
    )
    parser.add_argument(
        "--label_radius",
        type=float,
        default=32.0,
        help="Radius of target label exponential heatmap in pixels (default: 32.0)."
    )
    parser.add_argument(
        "--proc_size",
        type=int,
        default=800,
        help="Resolution for feature extraction and keypoint matching (default: 800)."
    )
    parser.add_argument(
        "--ransac_thresh",
        type=float,
        default=5.0,
        help="RANSAC reprojection error threshold in pixels (default: 5.0)."
    )
    parser.add_argument(
        "--target_min_motion",
        type=float,
        default=3.0,
        help="Minimum required motion velocity in pixels for target (default: 3.0)."
    )
    parser.add_argument(
        "--min_texture_std",
        type=float,
        default=3.0,
        help="Minimum local patch standard deviation to avoid low-contrast areas (default: 3.0)."
    )
    parser.add_argument(
        "--min_ncc",
        type=float,
        default=0.75,
        help="Minimum NCC patch similarity to filter matches (default: 0.75)."
    )
    parser.add_argument(
        "--temporal_step_seconds",
        type=float,
        default=1.0,
        help="Temporal sweep step in seconds between chronological seeks (default: 1.0)."
    )
    parser.add_argument(
        "--frame_gap_seconds",
        type=float,
        default=1.0,
        help="Temporal gap in seconds between hist and curr frames (default: 1.0)."
    )
    parser.add_argument(
        "--max_consecutive_failures",
        type=int,
        default=10,
        help="Maximum consecutive failures before skipping a video (default: 10)."
    )
    
    # Load default configs
    config_path = os.path.join(os.path.dirname(__file__), "pipeline_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                valid_keys = {action.dest for action in parser._actions if action.dest != 'help'}
                filtered_config = {k: v for k, v in config.items() if k in valid_keys}
                parser.set_defaults(**filtered_config)
        except Exception as e:
            print(f"Warning: Could not load pipeline_config.json ({e}). Using defaults.")
            
    args = parser.parse_args()
    
    if not os.path.isdir(args.videos_dir):
        print(f"Error: Directory '{args.videos_dir}' does not exist.")
        sys.exit(1)
        
    print("[*] Crawling for video files...")
    video_paths = find_video_files(args.videos_dir)
    if not video_paths:
        print(f"Error: No video files found in directory: {args.videos_dir}")
        sys.exit(1)
        
    print(f"[+] Found {len(video_paths)} valid videos.")
    
    asift_matcher = ASIFTMatcher()
    
    if args.visualize:
        visualize_pipeline(args, video_paths, asift_matcher)
    else:
        run_two_stage_pipeline(args, video_paths, asift_matcher)

if __name__ == "__main__":
    main()
