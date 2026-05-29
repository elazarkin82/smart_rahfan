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

def detect_and_match_single_type(f_hist, f_curr, feature_type, asift_matcher=None, ratio=0.85, min_inliers=6, ransac_thresh=5.0, min_motion=3.0, min_texture_std=3.0, proc_size=800, min_ncc=0.75):
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

    # 3. Apply Multi-Stage Geometric & Texture Filters
    for m in valid_matches:
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
    for idx, is_inlier in enumerate(mask_flat):
        if is_inlier:
            # Map coordinates back into normalized [0, 1] grid relative to proc_size
            h_norm = [pts_hist[idx][0] / proc_size, pts_hist[idx][1] / proc_size]
            c_norm = [pts_curr[idx][0] / proc_size, pts_curr[idx][1] / proc_size]
            inlier_paths.append({
                "hist": h_norm,
                "curr": c_norm
            })

    return inlier_paths

def match_features_doublet(f_hist, f_curr, ratio=0.85, min_inliers=6, ransac_thresh=5.0, min_motion=3.0, min_texture_std=3.0, proc_size=800, min_ncc=0.75, asift_matcher=None):
    """
    Executes and aggregates feature matches across multiple detectors (ASIFT, SURF, AKAZE).
    Keeps each matching metric distinct, and then pools verified moving paths together.
    """
    aggregated_paths = []
    
    # 1. ASIFT matching
    paths_asift = detect_and_match_single_type(
        f_hist, f_curr, 'asift', asift_matcher,
        ratio, min_inliers, ransac_thresh, min_motion, min_texture_std, proc_size, min_ncc
    )
    aggregated_paths.extend(paths_asift)

    # 2. SURF matching (graceful fallback inside function)
    paths_surf = detect_and_match_single_type(
        f_hist, f_curr, 'surf', None,
        ratio, min_inliers, ransac_thresh, min_motion, min_texture_std, proc_size, min_ncc
    )
    aggregated_paths.extend(paths_surf)

    # 3. AKAZE matching
    paths_akaze = detect_and_match_single_type(
        f_hist, f_curr, 'akaze', None,
        ratio, min_inliers, ransac_thresh, min_motion, min_texture_std, proc_size, min_ncc
    )
    aggregated_paths.extend(paths_akaze)

    # Dedup paths that are extremely close in coordinates
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
# Sequence Generation Stage
# =====================================================================

def is_path_in_inner_2_3(path):
    hx, hy = path["hist"]
    cx, cy = path["curr"]
    margin = 1.0 / 6.0
    return (margin <= hx <= 1.0 - margin and margin <= hy <= 1.0 - margin and
            margin <= cx <= 1.0 - margin and margin <= cy <= 1.0 - margin)

def run_two_stage_pipeline(args, video_paths, asift_matcher):
    print("\n===========================================================")
    print("=== TargetTracker3-Lite High-Throughput Dataset Pipe ===")
    print("===========================================================\n")
    sys.stdout.flush()
    
    # -------------------------------------------------------------------------
    # STAGE 1: Sequential Extraction of Symmetric 2-Frame Doublets
    # -------------------------------------------------------------------------
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
    
    # Filtering out already completed files for clean resume
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
            
        # Frame step corresponds to ~1.0 seconds seek sweeps
        frame_step = int(round(args.temporal_step_seconds * fps))
        gap_frames = int(round(fps)) # 1.0 second offset between hist and curr
        
        video_sample_count = 0
        consecutive_failures = 0
        current_frame_idx = 0
        max_start = total_frames - int(fps * 1.5)
        
        t_start_video = time.time()
        doublets_processed = 0
        successful_frames = 0
        
        while current_frame_idx <= max_start:
            # Seek and read hist frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame_hist_raw = cap.read()
            if not ret or frame_hist_raw is None:
                break
                
            # Seek and read curr frame (1.0 second after)
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
            
            # Match doublets across ASIFT, SURF, and AKAZE
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
                    if is_path_in_inner_2_3(path_candidate):
                        h_pt = path_candidate["hist"]
                        c_pt = path_candidate["curr"]
                        
                        # Generate Continuous Sharp Exponential Cone Mask
                        hist_mask = generate_exponential_mask(
                            h_pt, size=256, sigma=args.mask_sigma
                        )
                        
                        f_hist_norm = f_hist_256.astype(np.float32) / 255.0
                        f_curr_norm = f_curr_256.astype(np.float32) / 255.0
                        
                        # Pack inputs symmetrically: 
                        # hist_frame = (256, 256, 2) [Image, Mask]
                        # curr_frame = (256, 256, 1) [Image only]
                        hist_frame = np.stack([f_hist_norm, hist_mask[:, :, 0]], axis=-1)
                        curr_frame = np.expand_dims(f_curr_norm, axis=-1)
                        
                        # Generate Continuous Sharp Exponential Target Heatmap
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
            
            # Print live high-fidelity HUD console
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
            
            # Memory cleaning
            del f_hist_full, f_hist_256, f_hist_proc
            del f_curr_full, f_curr_256, f_curr_proc
            del frame_hist_raw, frame_curr_raw
            del inlier_paths
            gc.collect()
            
            if consecutive_failures >= args.max_consecutive_failures:
                print(f"\n[WARNING] Consecutive failures threshold reached ({consecutive_failures}). Skipping video.")
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
        
    # -------------------------------------------------------------------------
    # STAGE 2: Shuffled i.i.d. Batch Packaging
    # -------------------------------------------------------------------------
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
    run_two_stage_pipeline(args, video_paths, asift_matcher)

if __name__ == "__main__":
    main()
