"""
ingestor_aicity.py
AI City Track (2021) Track 4 - Anomaly Event Ingestor

Checked:
- Frame-index sync 
- jump decoding foundation

Confirmed from zip inspection
Zip root folder : AIC21-Track4-Anomaly-Detection/
Train videos : aic21-track4-train-data/
Annotation file : train-anomaly-results.txt
Has header row: NO
Columns : video_id start_frame end_frame
Class column : ABSENT
Video filenames : {video_id}.mp4
Frame basis : direct integer frame numbers (not seconds)

Block D class handling:
No class column --> Events are assigned anomaly_class_name = "anomaly" so all events have used_in_block_d = True
Anomaly class map not applied during row parsing IN THIS FILE - Retained in config for future use

DESIGN DECISIONS:
- Paths built from DATASET_ROOT environment variable only.
$env:DATASET_ROOT = 'D:\\datasets'
%env DATASET_ROOT = .../CST3990/datasets
- All subpaths, column indices & class config read from configs/config_day3.yaml
- Executed as: python -m src.ingestor_aicity from repo root.
- All frame indices validated against video bounds before saving
- "anomaly_class_name" is an event-level label only.

"""

import os
import json
import zipfile
import cv2
import shutil
import logging
import subprocess
import yaml
from pathlib import Path
from src.seed_control import set_all_seeds

set_all_seeds()

logging.basicConfig(
    level=logging.INFO,
    # Each line shows: timestamp, severity level, message
    format="%(asctime)s [%(levelname)s] %(message)s"
    
)

log = logging.getLogger(__name__)

# CONFIG LOADED

def load_config() -> dict:
    config_path = os.path.join("configs", "config_day3.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Create configs/config_day3.yaml before running.")

    with open(config_path) as f:
        return yaml.safe_load(f)
    
# PATH RESOLUTION

def resolve_aicity_paths(cfg: dict) -> dict:
    """
    Build all AI City paths from DATASET_ROOT environment

    Subfolders are split on '/'

    """
    root = os.environ.get("DATASET_ROOT", "").strip()
    if not root:
        raise EnvironmentError(
            "DATASET_ROOT is not set-\n"
            "Windows : $env:DATASET_ROOT = 'D: \\datasets'\n"
            "Colab : %env DATASET_ROOT=/content/drive/MyDrive/CST3990/datasets"

        )
    ai = cfg["aicity"]
    aicity_root = os.path.join(root, *ai["subfolder"].split("/"))
    videos_dir = os.path.join(aicity_root, *ai["videos_subfolder"].split("/"))
    anno_path = os.path.join(aicity_root, ai["annotation_filename"])

    return {
        "aicity_root": aicity_root,
        "videos_dir": videos_dir,
        "anno_path": anno_path 
    }

# OUTPUT DIRECTORIES

LOGS_DIR = "logs"
EVENTS_JSON = os.path.join(LOGS_DIR, "events.json")
SYNC_CHECK_DIR =  os.path.join(LOGS_DIR, "day3_sync_checks")
QC_DIR = os.path.join(LOGS_DIR,"day3_qc")
ZIP_MANIFEST = os.path.join(LOGS_DIR, "day3_zip_manifest.json")

for _d in [LOGS_DIR, SYNC_CHECK_DIR, QC_DIR]:
    os.makedirs(_d, exist_ok=True)

# STORAGE GUARD

def check_disk_space(path: str, min_gb: float) -> None:
    """
    Abort if free space on the drive containing path is below min_gb
    Uses os.path.splitdrive to find the Windows drive root safely.
    """

    drive = os.path.splitdrive(path)[0]
    check_root = (drive + os.sep) if drive else path
    _, _, free = shutil.disk_usage(check_root)
    free_gb = free / ( 1024**3)
    log.info(
        f"Disk check - Free on '{check_root}': {free_gb:.1f} GB "
        f"(required: {min_gb} GB)"
    )
    
    if free_gb < min_gb:
        raise RuntimeError(
            f"Insufficient disk space: {free_gb:.1f} GB free. "
            f"Need at least {min_gb} GB. Free up space before proceeding. "
        )
    
# ZIP MANIFEST

def write_zip_manifest(zip_path: str, max_entries: int) -> None:
    """
    Writing a JSON audit log of zip entry names

    Recording two separate counts:
        total_entries_in_zip : full entry count from zipfile metadata
        entries_logged : number actually written (capped at max_entries)

    """
    if not os.path.isfile(zip_path):
        log.info(f"Zip not found for manifest ({zip_path}) - skipping.")
        return
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            all_names = [info.filename for info in zf.infolist()]
            total_count = len(all_names)
            logged = all_names[:max_entries]

        manifest = {
            "zip_path": zip_path,
            "total_entries_in_zip": total_count,
            "entries_logged": len(logged),
            "max_entries_cap": max_entries,
            "truncated": total_count > max_entries,
            "entries": logged,
            "student_id": "M01014463",
            "note": (
                "Zip manifest written for audit reference."
                "Full entry count : total_entriess_in_zip."
                "Capped sample: entries_logged."
                "Raw zip is not committed to Github"
            ),
        }
        with open(ZIP_MANIFEST, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info(
            f"Zip manifest → {ZIP_MANIFEST} "
            f"({len(logged)} of {total_count} entries logged)"
        )
    except Exception as exc:
        log.warning(f"Could not write zip manifest: {exc}")

# VIDEO PROBE WITH CODE FALLBACK

def probe_video(video_path: str) -> dict:
    """
    Read video metadata via OpenCV.
    Fallback to ffprobe if OpenCV returns frame_count == 0
    (codec incompatibility, common with some AI City mp4 encodings in Colab)

    Returns : fps, frame_count, duration_sec, width, height, codec_source.
    Raises RuntimeError if both methods fail - caller skips the video.  
    """

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or fps > 120:
        raise ValueError(
            f"Suspicious FPS={fps} for {os.path.basename(video_path)}. "
            f"Check file integrity. Expected range: 1-120."
        )
    codec_source = "opencv"

    if frame_count == 0:
        log.warning(
            f"OpenCV frame_count=0 for {os.path.basename(video_path)}. "
            f"Attempting ffprobe fallback."
        )
        # Calls helper function
        frame_count, duration_sec = _ffprobe_frame_count(video_path, fps)
        codec_source = "ffprobe_fallback"
    else:
        duration_sec = round(frame_count / fps, 4)
    if frame_count == 0:
        raise RuntimeError(
            f"Cannot determine frame count for {video_path}.\n"
            f"Both OpenCV and ffprobe failed.\n"
            f"On Colab: !apt-get install -y ffmpeg\n"
            f"On Windows: install ffmpeg and add to PATH."
        )
    return {
        "fps": round(fps,4),
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
        "codec_source": codec_source,
    }
# Estimating frame count
def _ffprobe_frame_count(video_path: str, fps: float) -> tuple:
    try:
        # External command run on ffprobe
        result = subprocess.run(
            ["ffprobe",
              "-v", "error", 
              "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1,"
                "video_path"],
            capture_output=True, text=True, timeout=30
        )
        duration_sec = float(result.stdout.strip())
        frame_count = round(duration_sec * fps)
        log.info(
            f" ffprobe: duration={duration_sec:.2f}s → "
            f"frames={frame_count}"
        )
        return frame_count, round(duration_sec, 4)
    except Exception as exc:
        log.error(f"ffprobe failed: {exc}")
        return 0, 0.0

# FRAME BOUNDS VALIDATION (From annotated file)

def validate_and_clamp_frames(
        start_frame: int, end_frame: int,
        total_frames: int, event_id: str
) -> tuple:
    """
    Validate frame indices against video bounds.
    Returns (start_frame, end_frame, valid: bool).

    Rules:
        start_frame < 0    → skip event (return False)
        end_frame < start_frame    → skip event (return False)
        end_frame > total -1     → clamp to total-1, keep event(warn)

    """
    # NOTE : Frame indexing starts at 0.
    # Checks whether event begins before video starts
    if start_frame < 0:
        log.warning(
            f" [BOUNDS] {event_id}: start_frame={start_frame} < 0. "
            f"Skipping. Check annotation or frame index basis."
        )
        return start_frame, end_frame, False
    # Checks logical consistency
    if end_frame < start_frame:
        log.warning(
            f"[BOUNDS] {event_id}: end_frame={end_frame} < "
            f"start_frame={start_frame}. Skipping."
        ) 
        return start_frame, end_frame, False
    # Checks whether even goes beyond video length
    if end_frame > total_frames - 1:
        log.warning(
            f"[BOUNDS] {event_id}: end_frame={end_frame} clamped to "
            f"{total_frames -1} (video total_frames={total_frames})."
        )
        end_frame = total_frames - 1

    return start_frame, end_frame, True

# TWO-READ FRAME SYNC CHECK

def verify_frame_sync(
    video_path: str, annotation_frame_id: int,
    event_id: str, offset_tolerance: int
) -> dict:

    """
    After cap.set(CAP_PROP_POS_FRAMES, N), OpenCV's position reporting can reflect the next frame to be decpded rather 
    than the frame jusr set. 
    Recording both:
        pos_before_reas: position after seek, before cap.read()
        pos_after_read : position after cap.read() - redahead shift
    
    Sync pass/fail is evaluated on pos_before_read vs annotation_frame_id.

    readahead_shift is logged for diagnosis but does not affect pass/fail.

    Saved JPEG filename encodes bothr requested & actual position => Mismatch visible without opening

    Sync failure => Annotation may use a different frame indexing basis.

    NOTE : Inspect saved images to confirm scene content corresponds to expectd anomaly testing

    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, annotation_frame_id)
    pos_before_read = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ret, frame = cap.read()
    pos_after_read = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cap.release()

    result = {
        "event_id": event_id,
        "video": os.path.basename(video_path),
        "annotation_frame_id": annotation_frame_id,
        "pos_before_read": pos_before_read,
        "pos_after_read": pos_after_read,
        "offset_before_read": pos_before_read - annotation_frame_id,
        "readahead_shift": pos_after_read - pos_before_read,
        "frame_saved": False,
        "sync_ok": False,
        "note": "",
    }
    if not ret or frame is None:
        result["note"] = (
            "Frame could not be read. Check video file integrity. "
            "This video will produce unreliable sync results."
        )
        log.warning(
            f"[SYNC FAIL] {event_id}: cannot read frame "
            f"{annotation_frame_id} from {os.path.basename(video_path)}"
        )
        return result

    #Filename encodes both requested and actual frame - mismatch visible
    out_name = (
        f"sync_{event_id}"
        f"req{annotation_frame_id:06d}"
        f"got{pos_before_read:06d}.jpg"
    )
    out_path = os.path.join(SYNC_CHECK_DIR , out_name)
    cv2.imwrite(out_path, frame)
    result["frame_saved"] = True
    result["frame_path"] = out_path

    offset = abs(result["offset_before_read"])
    result["sync_ok"] = (offset <= offset_tolerance)

    if result["sync_ok"]:
        log.info(
            f" [SYNC OK] {event_id} | "
            f"req={annotation_frame_id} got={pos_before_read} "
            f"offset={result['offset_before_read']} "
            f"readahead={result['readahead_shift']}"
        )
    else:
        result["note"] = (
            f"Offset {result['offset_before_read']} exceeds tolerance "
            f"{offset_tolerance}. Possible 0-based vs 1-based mismatch. "
            f"Inspect imahe: {out_name}"
        )
        log.error(
            f"[SYNC FAIL] {event_id} | "
            f"req={annotation_frame_id} got={pos_before_read} "
            f"OFFSET={result['offset_before_read']} "
            f"Inspect {out_name} before Day 4."
        )
    return result

# QC FRAME SAVE
def save_qc_frame(
        video_path: str, start_frame: int,
        event_id: str, fps: float
) -> str:
    """
    Saving annotated frame at anomaly start for visual QC.
    NOTE : Run for first event of each anomaly class only (config-controlled).
    """

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        log.warning(f"[QC] Cannot read QC frame for {event_id}")
        return ""

    label = (
        f"Day 3 QC | {event_id} | "
        f"frame={start_frame} | fps={fps:.2f}"
    )
    cv2.putText(
        frame, label, (10,40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255, 0), 2
    )
    out_path = os.path.join(QC_DIR, f"qc_{event_id}.jpg")
    cv2.imwrite(out_path, frame)
    log.info(f" [QC] → {out_path}")

    return out_path

# ANNOTATUON PARSER
def parse_annotations(
        anno_path: str, videos_dir: str, cfg: dict
) -> tuple:
    
    """
    Parse train-anomaly-results.txt - AI City 2021 Track 4

    Confirmed annotation format:
        Space-separated, no header row, 3 columns:
        video_id start_frame end_frame
        No class column. AI rows represent anomaly events.

    Block D class handling:
    col_class is null in config => skip unknown_classes is irrelevant.
    All rows receive anomaly_class_name = default_class_name
    block_d_classes = ["anomaly] => used_in_block_d = True for all.
    anomaly_classes map is retained but not applied during parsing

    Multiple rows with same video_id are supported

    IMPORTANT:
    UA-DETRAC's CLASS_MAP does not interact with AI City

    Returns: events, sync_results, video_meta, skipped_rows
    skipped_rows contains one dict perskipped row : These are logged to events.json for examiner transparency. 
    """
    if not os.path.isfile(anno_path):
        raise FileNotFoundError(
            f"Annotation file not found: {anno_path}\n"
            f"Complete extraction process before running the ingestor.\n"
            f"Expected: AIC21-Track4-Anomaly-Detection/train-anomaly-results.txt"
        )
    ai_cfg = cfg["aicity"]
    col_vid = ai_cfg["col_video_id"]
    col_start = ai_cfg["col_start"]  
    col_end = ai_cfg["col_end"]
    col_class = ai_cfg.get("col_class")
    has_header = ai_cfg.get("has_header", False)
    vid_pattern = ai_cfg.get("video_filename_pattern", "{}.mp4")
    default_class = ai_cfg.get("default_class_name", "anomaly")
    skip_unknown = ai_cfg.get("skip_unknown_classes", False)
    frame_basis = ai_cfg.get("frame_index_basis", "direct_frame_number")
    offset_tol = cfg["sync"]["offset_tolerance_frames"]
    qc_per_class = cfg["sync"]["qc_frames_per_class"]
    block_d_classes = set(cfg.get("block_d_classes", []))

    # anomaly_classes map is NOT used for row parsing when col_class is None
    # Retained for reference only.

    anomaly_map = cfg.get("anomaly_class", {})

    events = []
    sync_results = []
    video_meta = {} # keyed by video filename
    synced_videos = set() #sync check completed
    qc_done = {}
    skipped_rows = [] # rows not converted to events
    raw_line_count = 0 # total non-empty lines parsed

    with open(anno_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if has_header and lines:
        log.info(f"Header row skipped: '{lines[0].strip()}'")
        lines = lines[1:]
    log.info(f"Non-header lines to parse: {len(lines)}")

    for line_idx, raw_line in enumerate(line):
        line = raw_line.strip()
        if not line:
            continue
        raw_line_count += 1
        
        parts = line.split()

        # Determine minimum required columns
        required_cols = max(col_vid, col_start, col_end) + 1
        if len(parts) < required_cols:
            reason = f"too few columns: got {len(parts)}, need {required_cols}"
            log.warning(f" Row {line_idx}: {reason}. Line: '{line}'")
            skipped_rows.append({"row": line_idx, "line":line, "reason": reason})
            continue

        # Parse video ID
        try:
            vid_int = int(parts[col_vid])
        except ValueError:
            reason = f"non-integer video_id='{parts[col_vid]}'"
            log.warning(f" Row {line_idx}: {reason}. Skipping")
            skipped_rows.append({"row": line_idx, "line": line, "reason": reason})
            continue

        # Build video filename
        video_fname = vid_pattern.format(vid_int)
        video_path = os.path.join(videos_dir, video_fname)

        if not os.path.isfile(video_path):
            reason = (
                f"video not found: {video_fname}. "
                f"Only 10 videos were extracted."
                f"Additional videos are extracted on day 13 for Block D."
            )
            log.warning(f" Row {line_idx}: {reason}")
            skipped_rows.append({"row": line_idx, "line": line, "reason": reason})
            continue
        
        # Probe video metadata - once per filename
        if video_fname not in video_meta:
            try:
                meta = probe_video(video_path)
                video_meta[video_fname] = meta
                log.info(
                    f" Probed {video_fname}: fps={meta['fps']} "
                    f"frames={meta['frame_count']} "
                    f"duration={meta['duration_sec']}s "
                    f"({meta['codec_source']})"
                )
            except (ValueError, RuntimeError) as exc:
                reason = f"video probe failed: {exc}"
                log.error(f" Row {line_idx}: {reason}")
                skipped_rows.append({"row": line_idx, "line": line, "reason": reason})
                # Mark video as failed so subsequent rows skip it without re-probing
                video_meta[video_fname] = None
                continue

        meta = video_meta[video_fname]
        if meta is None:
            # Previously failed probe - skip silently
            skipped_rows.append({
                "row": line_idx, "line": line,
                "reason": f"video probe previously failed: {video_fname}"
            })
            continue

        # Parse frame indices - direct integers, no fps conversion
        try:
            start_frame = int(parts[col_start])
            end_frame = int(parts[col_end])
        except ValueError as exc:
            reason = f"bad frame index: {exc}"
            log.warning(f" Row {line_idx}: {reason}. Skipping.")
            skipped_rows.append({"row": line_idx, "line": line, "reason": reason})
            continue

        # Determine anomaly class
        # col_class is None (confirmed) - use default for all rows
        if col_class is not None and col_class < len(parts):
            class_raw = parts[col_class].strip()
            class_name = anomaly_map.get(class_raw)
            if class_name is None:
                if skip_unknown:
                    reason = f"unknown class='{class_raw}', skip_unknown_classes=true"
                    log.warning(f" Row {line_idx}: {reason}. Skipping.")
                    skipped_rows.append({"row": line_idx, "line": line, "reason": reason})
                    continue
                else:
                    class_name = f"unknown_{class_raw}"
        else:
            # No class column -assign default
            class_raw = None
            class_name = default_class

        in_block_d = class_name in block_d_classes

        # Include line_idx in event_id to distinguish duplicate video_id rows
        event_id = f"ev{line_idx:04d}_{vid_int}_{class_name}"

        # Frame bounds validation

        total_frames = meta["frame_count"]
        start_frame, end_frame, valid = validate_and_clamp_frames(
            start_frame, end_frame, total_frames, event_id
        )
        if not valid:
            skipped_rows.append({
                "row": line_idx, "line": line,
                "reason": "invalid frame bounds after validation"
            })
            continue
        # sync check, first check per video only
        is_first_for_video = video_fname not in synced_videos
        if is_first_for_video:
            sync_r = verify_frame_sync(
                video_path, start_frame, event_id, offset_tol
            )
            sync_results.append(sync_r)
            synced_videos.add(video_fname)
            syn_outcome = sync_r["sync_ok"]
        else:
            # Inherit sync outcome from the first check for this video
            prior = next(
                (s for s in sync_results
                if s["video"] == video_fname), None
            )
            sync_outcome = prior["sync_ok"] if prior else None

        #QC frame - first N events per class
        qc_done.setdefault(class_name, 0)
        if qc_done[class_name] < qc_per_class:
            save_qc_frame(video_path, start_frame, event_id, ["fps"])
            qc_done[class_name] += 1
                
        # Build event record
        event = {
            # Identity
            "event_id": event_id,
            "video_id": vid_int,
            "video_file": video_fname,
            # Video metadata 
            "original_fps": meta["fps"],
            "total_frames": total_frames,
            "duration_sec": meta["duration_sec"],
            "width": meta["width"],
            "height": meta["height"],
            "codec_source": meta["codec_source"],
            # Temporal coordinates - direct frame numbers
            "anomaly_start_frame": start_frame,
            "anomaly_end_frame": end_frame,
            "frame_index_basis": frame_basis,
            # Anomaly classification
            "anomaly_class_raw": class_raw,
            "anomaly_class_name": class_name,
            "used_in_block_d": in_block_d,
            # Sync check outcome
            "sync_checked_this_event": is_first_for_video,
            "sync_ok": sync_outcome, 

        }
        events.append(event)

    # SUMMARY
    log.info(f"Annotation rows processed (non-empty): {raw_line_count}")
    log.info(f" Valid events produced: {len(events)}")
    log.info(f" Rows skipped: {len(skipped_rows)}")
    log.info(f" Block D events: {sum(1 for e in event if e ['used_in_block_d'])}")
    log.info(f" Unique videos probed: {len([v for v in video_meta if video_meta[v] is not None])}")

    # Save snyc results
    sync_path = os.path.join(LOGS_DIR, "day3_frame_sync_results.json")
    with open(sync_path, "w") as f:
        json.dump(sync_results, f, indent=2)
    log.info(f"Sync results → {sync_path}")

    failed_syncs = [s for s in sync_results if not s["sync_ok"]]
    if failed_syncs:
        log.error(
            f"{len(failed_syncs)} sync check(s) FAILED.\n"
            f"DO NOT proceed.\n"
            f"Inspect images in {SYNC_CHECK_DIR}.\n"
            f"Check whether annotations uses 0-based or 1-based frame numbers."
        )
    return events, sync_results, video_meta, skipped_rows, raw_line_count
# MAIN
def run_aicity_ingestor():
    log.info("=" * 60)
    log.info("Day 3 - AI City Track 4 (2021) Ingestor")
    log.info("=" * 60)
    
    cfg = load_config()
    paths = resolve_aicity_paths(cfg)

    log.info(f"DATASET_ROOT: {os.envirom.get('DATASET_ROOT', 'NOT SET')}")
    log.info(f"AI City root: {paths['aicity_root']}")
    log.info(f"Videos dir: {paths['videos_dir']}")
    log.info(f"Annotation file: {paths['anno_path']}")

    check_disk_space(paths["aicity_root"], cfg["aicity"]["min_free_gb"])

    if not os.path.isdir(paths["aicity_root"]):
        log.error(
            f"AI City root not found: {paths['aicity_root']}\n"
            f"Complete extraction before running the ingestor."
        )
        return
    #Zip manifest - find  zip in ai_city if still present
    ai_city_dir = os.path.join(
        os.environ.get("DATASET_ROOT", ""), "ai_city"
    )
    zip_candidates = list(Path(ai_city_dir).glob("*.zip")) if os.path.isdir(ai_city_dir) else []
    if zip_candidates:
        write_zip_manifest(
            str(zip_candidates[0]),
            max_entries=cfg["zip_manifest"]["max_entries_to_log"]
        )
    else:
        log.info("No zip file found for manifest - skipping(already extracted)")
    
    # Run parser
    events, sync_results, video_meta, skipped, raw_line_count = parse_annotations(
        paths["anno_path"],
        paths["videos_dir"],
        cfg,
    )

    if not events:
        log.warning(
            "No events parsed. Possible causes:\n"
            "- Extraction not complete\n"
            "- config_day3.yaml column indices wrong\n"
            "- video files missing for all annotated videos IDs\n"
            "Re-check and re-run"
        )
        return
    # WRITE events.json
    output = {
        "schema_version": "1.0",
        "dataset": "AI City Challenge 2021 Track 4",
        "annotation_file": "train-anomaly-results.txt",
        "frame_index_basis": cfg["aicity"]["frame_index_basis"],
        "format_confirmed": (
            "Space-separated, no header, 3 columns: video_id start_frame end_frame. "
            "Confirmed by inspection of zip contents. "
        ),
        "block_d_class_note": (
            "No class column in annotation file."
            "All events assigned anomaly_class_name='anomaly'. "
            "block_d_classes=['anomaly] so used_in_block_d=True for all valid events. "
            "anomaly_classes map (stall/crash/etc.) retained in config for future use "
            "if per-event class labels are discovered."
        ),
        "anomaly_schema_note": (
            "anomaly_class_name is an event-level label. "
            "Complelety separate from UA-DETRAC object-detection CLASS_MAP."
        ),
        "sync_check_note": "Added two-read method. See day3_frame_sync_results.json.",
        "github_note": "events.json committed to Github. Raw videos stay on SSD only.",
        "block_d_classes": sorted(cfg.get("block_d_classes", [])),
        # Counts - transparent reporting
        "annotation_rows_processed": raw_line_count,
        "valid_events_produced": len(events),
        "skipped_rows_count": len(skipped),
        "skipped_rows": skipped,
        "block_d_events": sum(1 for e in events if e["used_in_block_d"]),
        "unique_videos_probed": len([v for v in video_meta if video_meta[v] is not None]),
        "events": events,
    }
    with open(EVENTS_JSON, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"events.json → {EVENTS_JSON} ")
    log.info(f"Annotation rows processed: {raw_line_count}")
    log.info(f"Valid events produced: {len(events)}")
    log.info(f"Skipped rows: {len(skipped)}")
    log.info(f"Block D events: {output['block_d_events']}")

    # EXIT CRITERIA SELF-CHECK
    failed_syncs = [s for s in sync_results if not s["sync_ok"]]

    log.info("\n" + "=" * 60)
    log.info("DAY 3 EXIT CRITERIA - AUTO CHECKED")
    log.info("=" * 60)

    auto_checks= [
        (
            os.path.isfile(EVENTS_JSON),
            "events.json written"
        ),
        (
            os.path.isfile(os.path.join(LOGS_DIR, "day3_frame_sync_results.json")),
            "day3_frame_sync_results.json written"
        ),
        (
            len(failed_syncs) ==0,
            f"All sync checks passed ({len(failed_syncs)} failure(s))"
        ),
        (
            bool(os.listdir(SYNC_CHECK_DIR)),
            "Sync check images exist in day3_sync_checks/"
        ),
        (
            bool(os.listdir(QC_DIR)),
            "QC frames exist in day3_qc/"
        ),
        (
            output["valid_events_produced"] > 0,
            f"Valid events produced: {output['valid_events_produced']} "
            f"(From {raw_line_count} rows, {len(skipped)} justified skips)"
        ),
        (
            output["block_d_events"] == output["valid_events_produced"],
            "All valid events are Block D events (used_in_block_d=True)"
        ),
    ]
    
    for passed, label in auto_checks:
        log.info(f" [{'PASS' if passed else 'FAIL'}] {label}")
    log.info("")
    log.info(" MANUAL CHECKS - Confirm personally:")
    log.info(" [] Sync images visually inspected in dya3_sync_checks/")
    log.info(" Open imaages and confirm scene matches annotation frame.")
    log.info("[] All file sources pushed to Github")
    log.info("[] Notebook saved to drive")
    log.info("[] Raw AI City videos confirmed NOT committed to Github")
    log.info("=" * 60)
    log.info("Day 3 AI ")
    if failed_syncs:
        log.error("SYNC FAILURES PRESENT - DO NOT PROCEED: ")
if __name__ == "__main__":
    run_aicity_ingestor()

