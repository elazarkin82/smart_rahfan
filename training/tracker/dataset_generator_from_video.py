#!/usr/bin/env python3
import os
import sys
import random
import pickle
import argparse
import numpy as np
import cv2

# =====================================================================
# Heatmap and Circular Masking Helpers (Matching tracker_model2 exactly)
# =====================================================================

def apply_circle_mask(img_uint8, coords, radius):
    """
    Applies a binary circular mask to a grayscale image (0-255 uint8)
    centered around the normalized coordinate, blacking out the region outside.
    
    Args:
        img_uint8 (np.ndarray): Grayscale image array of shape (H, W).
        coords (list/tuple): Normalized [x, y] coordinates in [0, 1] range.
        radius (int): Radius of the circular mask in pixels.
        
    Returns:
        np.ndarray: Grayscale image with the mask applied.
    """
    h, w = img_uint8.shape[:2]
    x_px = int(coords[0] * w)
    y_px = int(coords[1] * h)
    
    mask = np.zeros_like(img_uint8)
    cv2.circle(mask, (x_px, y_px), radius, 255, -1)
    
    masked_img = cv2.bitwise_and(img_uint8, mask)
    return masked_img

def generate_gaussian_heatmap(coords, size=64, sigma=4.0):
    """
    Generates a 2D Gaussian heatmap centered around the normalized coordinate.
    
    Args:
        coords (list/tuple): Normalized [x, y] coordinate in [0, 1] range.
        size (int): Dimensions of the output heatmap grid (default: 64).
        sigma (float): Standard deviation of the Gaussian kernel (default: 4.0).
        
    Returns:
        np.ndarray: Gaussian heatmap array of shape (size, size, 1).
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

def match_sift_triplet(f_hist, f_prev, f_curr, ratio=0.75, min_inliers=8):
    """
    Runs SIFT keypoint detection and matches features across three frames
    (hist -> prev and prev -> curr). Fits a Fundamental Matrix via RANSAC
    to reject dynamic outliers and returns a list of verified inlier coordinate paths.
    
    Args:
        f_hist (np.ndarray): Hist grayscale frame (256x256).
        f_prev (np.ndarray): Prev grayscale frame (256x256).
        f_curr (np.ndarray): Curr grayscale frame (256x256).
        ratio (float): Lowe's ratio test threshold.
        min_inliers (int): Minimum required verified RANSAC inliers.
        
    Returns:
        dict: Containing matching paths or None if matching failed.
    """
    sift = cv2.SIFT_create()
    
    kp_hist, des_hist = sift.detectAndCompute(f_hist, None)
    kp_prev, des_prev = sift.detectAndCompute(f_prev, None)
    kp_curr, des_curr = sift.detectAndCompute(f_curr, None)
    
    if des_hist is None or des_prev is None or des_curr is None:
        return None
        
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
                
    # Find matching keypoint triplets (hist -> prev -> curr)
    pts_hist, pts_prev, pts_curr = [], [], []
    kp_triplets = []
    
    for idx_hist, idx_prev in hist_to_prev.items():
        if idx_prev in prev_to_curr:
            idx_curr = prev_to_curr[idx_prev]
            pts_hist.append(kp_hist[idx_hist].pt)
            pts_prev.append(kp_prev[idx_prev].pt)
            pts_curr.append(kp_curr[idx_curr].pt)
            kp_triplets.append((idx_hist, idx_prev, idx_curr))
            
    if len(pts_hist) < min_inliers:
        return None
        
    pts_hist = np.float32(pts_hist)
    pts_prev = np.float32(pts_prev)
    pts_curr = np.float32(pts_curr)
    
    # Fit Fundamental Matrices using RANSAC to verify epipolar constraints
    # (Removes dynamic moving objects like other cars and keeps the rigid background)
    F_12, mask_12 = cv2.findFundamentalMat(pts_hist, pts_prev, cv2.FM_RANSAC, 3.0)
    F_23, mask_23 = cv2.findFundamentalMat(pts_prev, pts_curr, cv2.FM_RANSAC, 3.0)
    
    if F_12 is None or F_23 is None or mask_12 is None or mask_23 is None:
        return None
        
    # Safely flatten RANSAC masks to 1D flat arrays to avoid array truth ambiguity
    m12_flat = mask_12.ravel()
    m23_flat = mask_23.ravel()
    
    inlier_indices = []
    for i in range(len(pts_hist)):
        if m12_flat[i] == 1 and m23_flat[i] == 1:
            inlier_indices.append(i)
            
    if len(inlier_indices) < min_inliers:
        return None
        
    # Construct verified normalized paths
    verified_paths = []
    for idx in inlier_indices:
        h_pt = pts_hist[idx]
        p_pt = pts_prev[idx]
        c_pt = pts_curr[idx]
        
        # Coordinates normalized to [0, 1] range based on 256x256 frame dims
        path = {
            "hist": [np.clip(h_pt[0] / 256.0, 0.0, 1.0), np.clip(h_pt[1] / 256.0, 0.0, 1.0)],
            "prev": [np.clip(p_pt[0] / 256.0, 0.0, 1.0), np.clip(p_pt[1] / 256.0, 0.0, 1.0)],
            "curr": [np.clip(c_pt[0] / 256.0, 0.0, 1.0), np.clip(c_pt[1] / 256.0, 0.0, 1.0)]
        }
        verified_paths.append(path)
        
    return {
        "paths": verified_paths,
        "kp_hist": kp_hist,
        "kp_prev": kp_prev,
        "kp_curr": kp_curr,
        "triplets": kp_triplets,
        "inliers": inlier_indices
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
    
    # Safely guard bounds to avoid empty slice crashes in custom displays
    y1 = max(0, org[1] - h - 4)
    y2 = min(img.shape[0], org[1] + baseline)
    x1 = max(0, org[0] - 4)
    x2 = min(img.shape[1], org[0] + w + 4)
    
    sub_img = img[y1:y2, x1:x2]
    if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
        rect = np.zeros_like(sub_img)
        rect[:] = 15  # Very dark grey
        cv2.addWeighted(sub_img, 0.3, rect, 0.7, 0, sub_img)
    
    # Draw neon text
    cv2.putText(img, label, org, font, scale, color, thickness, cv2.LINE_AA)

def render_dashboard(hist_m, prev_m, curr, hist_norm, prev_norm, curr_norm, sift_info=None):
    """
    Renders a stunning 4-panel dashboard containing masked historical and previous frames,
    the current unmasked frame overlaid with green target rings, and a SIFT tracking match diagram.
    """
    # 1. Convert grayscale panels to color (BGR)
    h_color = cv2.cvtColor(hist_m, cv2.COLOR_GRAY2BGR)
    p_color = cv2.cvtColor(prev_m, cv2.COLOR_GRAY2BGR)
    c_color = cv2.cvtColor(curr, cv2.COLOR_GRAY2BGR)
    
    # 2. Draw Target Dots and Mask Boundaries
    # Hist: Red dot
    hx, hy = int(hist_norm[0] * 256), int(hist_norm[1] * 256)
    cv2.circle(h_color, (hx, hy), 5, (0, 0, 255), -1)
    cv2.circle(h_color, (hx, hy), 128, (60, 60, 60), 1)
    
    # Prev: Blue dot
    px, py = int(prev_norm[0] * 256), int(prev_norm[1] * 256)
    cv2.circle(p_color, (px, py), 5, (255, 0, 0), -1)
    cv2.circle(p_color, (px, py), 50, (60, 60, 60), 1)
    
    # Curr: Neon green tracking rings
    cx, cy = int(curr_norm[0] * 256), int(curr_norm[1] * 256)
    cv2.circle(c_color, (cx, cy), 8, (0, 255, 0), 2)
    cv2.circle(c_color, (cx, cy), 2, (0, 255, 0), -1)
    
    # 3. Create the SIFT Debug panel
    if sift_info is not None:
        # Match diagram between f_prev (left) and f_curr (right)
        sift_panel = np.hstack([prev_m, curr])
        sift_panel_color = cv2.cvtColor(sift_panel, cv2.COLOR_GRAY2BGR)
        
        kp_prev = sift_info["kp_prev"]
        kp_curr = sift_info["kp_curr"]
        triplets = sift_info["triplets"]
        inliers = sift_info["inliers"]
        
        # Draw all verified inlier match lines in cyan
        for idx in inliers:
            trip = triplets[idx]
            pt_p = kp_prev[trip[1]].pt
            pt_c = kp_curr[trip[2]].pt
            
            p1 = (int(pt_p[0]), int(pt_p[1]))
            p2 = (int(pt_c[0] + 256), int(pt_c[1]))  # Shifted right by width
            
            # Glowing cyan line
            cv2.line(sift_panel_color, p1, p2, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(sift_panel_color, p1, 3, (255, 0, 255), -1)
            cv2.circle(sift_panel_color, p2, 3, (255, 0, 255), -1)
            
        s_color = cv2.resize(sift_panel_color, (256, 256), interpolation=cv2.INTER_AREA)
    else:
        # Stationary Hover case (no match triplets)
        s_color = cv2.cvtColor(prev_m, cv2.COLOR_GRAY2BGR)
        cv2.putText(s_color, "HOVER MODE", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(s_color, "Gimbal Jitter Active", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1, cv2.LINE_AA)
        
    # 4. Draw HUD Labels
    draw_hud_label(h_color, "HIST FRAME (MASK R=128)", (10, 240), (0, 0, 255))
    draw_hud_label(p_color, "PREV FRAME (MASK R=50)", (10, 240), (255, 0, 0))
    draw_hud_label(c_color, "CURR FRAME (GROUND TRUTH)", (10, 240), (0, 255, 0))
    draw_hud_label(s_color, "SIFT RANSAC TRACKING LINES", (10, 240), (255, 255, 0))
    
    # 5. Assemble Grid
    row1 = np.hstack([h_color, p_color])
    row2 = np.hstack([c_color, s_color])
    dashboard = np.vstack([row1, row2])
    
    # 6. Add Top HUD Dashboard header bar
    header_bar = np.zeros((35, 512, 3), dtype=np.uint8)
    cv2.putText(header_bar, "VIDEO DATASET GENERATOR - PREVIEW HUD", (10, 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header_bar, "[SPACE]: Next  |  [ESC/Q]: Exit", (295, 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
    final_output = np.vstack([header_bar, dashboard])
    return final_output

# =====================================================================
# Main Dataset Generation Loop
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generates training datasets for tracker_model2 directly from real driving/drone videos using SIFT RANSAC."
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
        default=0.75,
        help="Lowe's ratio threshold for filtering SIFT matches (default: 0.75)."
    )
    parser.add_argument(
        "--min_inliers",
        type=int,
        default=8,
        help="Minimum number of verified RANSAC inliers to accept a triplet (default: 8)."
    )
    parser.add_argument(
        "--hover_prob",
        type=float,
        default=0.15,
        help="Probability of selecting stationary hovering with gimbal camera shake (default: 0.15)."
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
        print("Controls: Press [SPACE] for next frame, [ESC] or [Q] to exit.\n")
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
        
        while len(hist_frames_batch) < args.batch_size:
            if args.visualize and sample_count >= args.num_of_samples:
                break
                
            # 1. Decide if this sample should simulate hovering (static frame + jitter)
            is_hover = random.random() < args.hover_prob
            
            random_video = random.choice(video_paths)
            cap = cv2.VideoCapture(random_video)
            
            if not cap.isOpened():
                continue
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 200:
                cap.release()
                continue
                
            # Define maximum starting index to prevent running out of frames
            max_start = total_frames - 150
            start_frame_idx = random.randint(0, max_start)
            
            # Performance Optimization: Seek to F_hist once, then grab frames sequentially
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
            ret, frame_hist_raw = cap.read()
            if not ret or frame_hist_raw is None:
                cap.release()
                continue
                
            # Safely convert raw frame to grayscale
            f_hist_full = to_grayscale(frame_hist_raw)
            f_hist_256 = cv2.resize(f_hist_full, (256, 256), interpolation=cv2.INTER_AREA)
            
            if is_hover:
                # =========================================================
                # Stationary Hover Scenario (Zero I/O seek, SIFT on 1 frame)
                # =========================================================
                cap.release()
                
                sift = cv2.SIFT_create()
                kp, _ = sift.detectAndCompute(f_hist_256, None)
                if not kp:
                    continue
                    
                # Pick a random keypoint as target
                target_kp = random.choice(kp)
                target_coords = [target_kp.pt[0] / 256.0, target_kp.pt[1] / 256.0]
                
                # Jitter frame 1 to get prev
                f_prev_256, prev_coords = simulate_hover_jitter(f_hist_256, target_coords)
                
                # Jitter frame 2 to get curr
                f_curr_256, curr_coords = simulate_hover_jitter(f_hist_256, target_coords)
                
                hist_coords = target_coords
                sift_match_debug = None
                
            else:
                # =========================================================
                # Real Drone / Driving Camera Translation (Consecutive Seek)
                # =========================================================
                # Fast forward skip (grab frames without heavy decoding)
                gap_k = random.randint(60, 110)
                for _ in range(gap_k - 1):
                    cap.grab()
                ret, frame_prev_raw = cap.read()
                
                if not ret or frame_prev_raw is None:
                    cap.release()
                    continue
                    
                f_prev_full = to_grayscale(frame_prev_raw)
                f_prev_256 = cv2.resize(f_prev_full, (256, 256), interpolation=cv2.INTER_AREA)
                
                # Skip to current frame (very small step, e.g. 2 frames)
                gap_d = 2
                for _ in range(gap_d - 1):
                    cap.grab()
                ret, frame_curr_raw = cap.read()
                cap.release()
                
                if not ret or frame_curr_raw is None:
                    continue
                    
                f_curr_full = to_grayscale(frame_curr_raw)
                f_curr_256 = cv2.resize(f_curr_full, (256, 256), interpolation=cv2.INTER_AREA)
                
                # 2. Keypoint Matching & Geometric RANSAC verification
                match_res = match_sift_triplet(
                    f_hist_256, f_prev_256, f_curr_256, 
                    ratio=args.ratio, min_inliers=args.min_inliers
                )
                
                if match_res is None:
                    continue
                    
                # Pick a random verified inlier landmark path
                selected_path = random.choice(match_res["paths"])
                
                hist_coords = selected_path["hist"]
                prev_coords = selected_path["prev"]
                curr_coords = selected_path["curr"]
                sift_match_debug = match_res
                
            # =========================================================
            # Common Sample Processing & Packaging
            # =========================================================
            
            # Apply circular masks centered on matching target landmarks
            f_hist_masked = apply_circle_mask(f_hist_256, hist_coords, radius=128)
            f_prev_masked = apply_circle_mask(f_prev_256, prev_coords, radius=50)
            
            # If visualize mode, display HUD preview immediately
            if args.visualize:
                dashboard = render_dashboard(
                    f_hist_masked, f_prev_masked, f_curr_256,
                    hist_coords, prev_coords, curr_coords,
                    sift_info=sift_match_debug
                )
                try:
                    cv2.imshow("Video Dataset Generator Debugger", dashboard)
                    key = cv2.waitKey(0) & 0xFF
                    if key == 27 or key == ord('q'):  # Esc or Q to quit
                        cv2.destroyAllWindows()
                        print("\nHUD Preview Mode exited by user.")
                        sys.exit(0)
                except cv2.error as e:
                    print(f"\nGUI Error: Could not render visualization window ({e}).")
                    print("This usually occurs in headless environments lacking X11 display forwarding.")
                    print("Please run without the '-v' / '--visualize' flag to export pickle files directly.")
                    sys.exit(1)
                    
                # Any other key (or space) loops to show next sample
                sample_count += 1
                print(f"Rendered sample {sample_count}/{args.num_of_samples} (Hover: {is_hover})")
                continue
                
            # Convert to target training formats (normalized float32, float32 heatmaps)
            hist_frame = np.expand_dims(f_hist_masked.astype(np.float32) / 255.0, axis=-1)
            prev_frame = np.expand_dims(f_prev_masked.astype(np.float32) / 255.0, axis=-1)
            curr_frame = np.expand_dims(f_curr_256.astype(np.float32) / 255.0, axis=-1)
            
            target_heatmap = generate_gaussian_heatmap(curr_coords, size=64, sigma=4.0)
            
            hist_frames_batch.append(hist_frame)
            prev_frames_batch.append(prev_frame)
            curr_frames_batch.append(curr_frame)
            target_heatmaps_batch.append(target_heatmap)
            
            hist_coords_batch.append(hist_coords)
            prev_coords_batch.append(prev_coords)
            curr_coords_batch.append(curr_coords)
            
            sample_count += 1
            
        if args.visualize:
            continue
            
        # Serialize batch to pickle matching tracker_model2 expectations exactly
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
