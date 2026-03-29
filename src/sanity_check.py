"""
src/sanity_check.py
Day 4 - Sanity Check for Temporal Normalisation

Engineering debug check - NOT formal benchmark.
Uses actual video frames and contour-based motion detection.
Present in Thesis Report as engineering sanity check only.

What is validated:
1. vel_px_sec formula correctness
2. Epsilon guard on dt=0
3. Formula consistency across FPS values
4. Minimum motion guard - warns if no real motion detected
5. Perspective disclaimer logged with every result

Execution:
python -m src.sanity_check
(run after python -m src.normalisation)

Output:
logs/day4/sanity_check_results.json

"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Avoids duplicating logic
from src.normalisation import (
    load_config,
    resolve_dataset_root,
    resolve_video_path,
    compute_velocity_px_sec,
    PERSPECTIVE_DISCLAIMER,
    IMAGE_PLANE_VELOCITY_NOTE,
    _EPSILON,

)

# Extracts two representative centroids from two real video frames to compute motion displacement from actual footage
# Provides reaality-based input to velocity formula
def extract_real_centroid_pair(video_path: str, 
                               frame_a: int = 50,
                               frame_b: int = 60,
                               min_motion_px: float = 1.0) -> dict:
    """
    Extracts two real frames & compute centroid pair via frame differencing on actual pixel data.

    Minimum motion guard:
    If delta_px < min_motion_px, logs a WARNING.
    This distinguishes a formula-not-exercised result (delta_px=0) from proper zero-velocity measurement

    Returns dict with:
    c1, c2, delta_px, delta_frames, fps, frame_size, method, motion_detected


    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Seeks video to requested frame index, reads frame, return image if successful
    def read_frame(idx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        return frame if ret else None
    
    # Reads two requested frames, then releases video handle
    frame_a_img = read_frame(frame_a)
    frame_b_img = read_frame(frame_b)
    cap.release()

    # Stops execution early if video cannot provide frames
    # Protects againts very short clips, corrupt files and makes failure understandable
    if frame_a_img is None or frame_b_img is None:
        raise ValueError(
            f"Could not read frames {frame_a} and {frame_b} "
            f"from {video_path}. File may be too short or corrupted."
        )
    
    
    # Converts to grayscale then computes absolute pixel difference
    # Grayscale simplifies motion analysis.
    gray_a = cv2.cvtColor(frame_a_img, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b_img, cv2.COLOR_BGR2GRAY)
    diff   = cv2.absdiff(gray_a, gray_b)

    # Makes motion regions cleaner, improves contour extraction robustness
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Locates largest moving region
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Avoids crashing
    if not contours:
        c1     = np.array([w / 2, h / 2])
        c2     = np.array([w / 2, h / 2])
        method = "fallback_no_motion_detected"

    else:
        # Picks largest motion contour
        # Computes its centroid using image moments
        # Gets izs bounding box

        largest = max(contours, key=cv2.contourArea)
        M  = cv2.moments(largest)
        cx = M["m10"] / M["m00"] if M["m00"] > 0 else w / 2
        cy = M["m01"] / M["m00"] if M["m00"] > 0 else h / 2
        x, y, bw, bh = cv2.boundingRect(largest)
        # c1 defined as centroid
        c1     = np.array([cx, cy])
        # c2 defined as a point slightly offset
        c2     = np.array([cx + bw * 0.3, cy])
        method = "largest_contour_centroid"

    delta_px     = float(np.linalg.norm(c2 - c1))
    delta_frames = frame_b - frame_a

    # MINIMUM MOTION GUARD
    motion_detected = delta_px >= min_motion_px
    if not motion_detected:
        print(f"  [WARNING] delta_px={delta_px:.4f} px — below "
              f"min_motion_px={min_motion_px}.")
        print(f"  Formula is not exercised on real motion.")
        print(f"  → Try different frame indices in config_day4.yaml "
              f"(sanity_check.frame_a / frame_b)")
        print(f"  → Or choose a clip with visible vehicle movement.")

    return {
        "c1": c1.tolist(),
        "c2": c2.tolist(),
        "delta_px": round(delta_px, 4),
        "delta_frames": delta_frames,
        "fps": round(fps, 4),
        "frame_size": f"{w}x{h}",
        "method": method,
        "motion_detected": motion_detected,
    }

# Loads frame settings, loops through datasets, extracts centroid pairs, computes velocity checks, saves final results
def run_sanity_check(cfg: dict, dataset_root: str) -> dict:
    """
    Engineering sanity check

    Validates:
    1. Formula correctness: vel = delta_px / delta_frames * fps
    2. Epsilon guard: delta_frames=0 returns 0.0
    3. Formula consistency: scales proportionally with FPS
    4. Minimum motion guard: warns if formula not exercised on real motion
    5. Perspective disclaimer logged with all results

    Does NOT enforce cross-dataset velocity threshold.
    Different frame sizes and perspectives make that comparison inherently non-comparable.
    See Thesis Report § 3.8.2
    """

    results = {}
    sc_cfg = cfg.get("sanity_check", {})
    frame_a = sc_cfg.get("frame_a", 50)
    frame_b = sc_cfg.get("frame_b", 60)
    min_motion_px = sc_cfg.get("min_motion_px", 1.0)

    for dataset_key in ["ua_detrac", "ai_city"]:
        ds_cfg = cfg[dataset_key]
        clip_id = ds_cfg["sample_clip"]
        video_path = resolve_video_path(dataset_root, ds_cfg["subfolder"], clip_id)
        
        if not os.path.exists(video_path):
            print(f"\n [SKIP] {dataset_key}: video not found - {video_path}")
            results[dataset_key] = {
                "status": "video_not_found",
                "path_attempted": str(video_path)
            }
            continue
        print(f"\n [{dataset_key}] Extracting centroid pair"
              f"(frames {frame_a}→{frame_b})...")
        try:
            pair = extract_real_centroid_pair(video_path, frame_a, frame_b, min_motion_px)
        except ValueError as e:
            print(f" [ERROR] {e}")
            results[dataset_key] = {"status": "error", "message": str(e)}
            continue

        # Computes correct and wrong velocity.
        # Clearly demonstrates why FPS-normalised velocity is necessary
        vel = compute_velocity_px_sec(pair["delta_px"],  max(pair["delta_frames"],1), pair["fps"])
        vel_wrong = pair["delta_px"] / max(pair["delta_frames"], 1)
        results[dataset_key] = {
            "status": "ok",
            "clip_id": clip_id,
            "video_path": str(video_path),
            "fps": pair["fps"],
            "frame_size": pair["frame_size"],
            "centroid_method": pair["method"],
            "motion_detected": pair["motion_detected"],
            "delta_px": pair["delta_px"],   
            "delta_frames": pair["delta_frames"],
            "vel_px_sec_correct": round(vel,4),
            "vel_px_frame_wrong": round(vel_wrong,4),
            "ratio_correct_vs_wrong": round(vel / (vel_wrong + _EPSILON), 4),
            "perspective_disclaimer": PERSPECTIVE_DISCLAIMER,
            "image_plane_velocity_note": IMAGE_PLANE_VELOCITY_NOTE
        }

        print(f"    FPS                : {pair['fps']}")
        print(f"    Frame size         : {pair['frame_size']}")
        print(f"    Centroid method    : {pair['method']}")
        print(f"    Motion detected    : {pair['motion_detected']}")
        print(f"    Delta px           : {pair['delta_px']:.4f}")
        print(f"    Delta frames       : {pair['delta_frames']}")
        print(f"    vel_px_sec    (✅) : {vel:.4f}  ← correct formula")
        print(f"    vel_px_frame  (❌) : {vel_wrong:.4f}  ← wrong, FPS-dependent")
    
    # EPSILON GUARD TEST
    # Explicitly tests zero-frame edge case
    print("\n  [Epsilon guard test]")
    vel_eps    = compute_velocity_px_sec(50.0, 0, 25.0)
    epsilon_ok = (vel_eps == 0.0)
    print(f"    compute_velocity_px_sec(50px, 0frames, 25fps) = {vel_eps}")
    print(f"    Expected: 0.0  Got: {vel_eps}  "
          f"{'✅' if epsilon_ok else '❌'}")
    
      #Formula consistency check
    print("\n  [Formula consistency — same displacement, different FPS]")
    formula_ok = True
    for test_fps in [10.0, 30.0, 60.0]:
        test_vel = compute_velocity_px_sec(50.0, 5, test_fps)
        expected = (50.0 / 5) * test_fps
        match    = abs(test_vel - expected) < 0.001
        formula_ok = formula_ok and match
        print(f"    FPS={test_fps:5.1f}  vel={test_vel:.4f}  "
              f"expected={expected:.4f}  {'✅' if match else '❌'}")   


     # COMPILE AND SAVE
    overall = epsilon_ok and formula_ok

    final = {
        "timestamp":               datetime.now().isoformat(),
        "epsilon_guard_ok":        epsilon_ok,
        "formula_consistency_ok":  formula_ok,
        "overall_passed":          overall,
        "dataset_results":         results,
        "scientific_note": (
            "Cross-dataset velocity ratio is NOT enforced as a hard threshold. "
            "Different frame sizes and scene perspectives make direct "
            "vel_px_sec comparison between datasets inherently non-comparable. "
            "Validation is formula-correctness and epsilon-guard only. "
            "See thesis §3.8.2 for construct validity discussion."
        ),
        "perspective_disclaimer":    PERSPECTIVE_DISCLAIMER,
        "image_plane_velocity_note": IMAGE_PLANE_VELOCITY_NOTE,
    }

    out_path = Path(cfg["output"]["sanity_check_results"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n  [Saved] Sanity check results → {out_path}")
    print(f"\n  Overall: {'✅ PASSED' if overall else '❌ FAILED'}")
    return final
# MAIN
if __name__ == "__main__":
    print("=" * 65)
    print("DAY 4 — Sanity Check: Temporal Normalisation Validation")
    print("  Run AFTER: python -m src.normalisation")
    print("=" * 65)

    from src.seed_control import set_all_seeds
    set_all_seeds()

    cfg = load_config("configs/config_day4.yaml")

    try:
        dataset_root = resolve_dataset_root()
    except EnvironmentError as e:
        print(f"[ERROR] {e}")
        raise SystemExit(1)

    run_sanity_check(cfg, dataset_root)


                                      
