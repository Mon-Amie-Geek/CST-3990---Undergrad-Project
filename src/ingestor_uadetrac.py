"""
ingestor_uadetrac.py
Fixes:
- Official train split, occlusion filter
- Runtime FPS adapted for image sequences
- Detector JSON schema frozen as reference
- Warmup frames stripped before writing
- Seed control at start of main

File : UA-DETRAC JPG image-sequence-to-YOLO ingestor
NOTE: To be run as module from project root (Eliminates import ambiguity):
python -m src.ingestor_uadetrac \
    --clips_master       configs/clips_master.txt\
    --images_dir         /path/to/raw\
    --annotations_dir    /path/to/annotations\
    --processed_dir      data/ua_detrac/processed\
    --logs_dir           logs

GITHUB/ DATA BOUNDARY (mandatory rule):
Raw dataset files are NEVER committed to Github. Only code, configs, lightweight logs and notebooks are version-controlled.

RERUN SAFETY:
Always run the cleanup/reset cell in Colab before re-running. This prevents stale files from contaminating splits across runs.

"""
# Finds and manages files/folders
import os
# For image reading
import cv2
# Saves structured information
import json
# Move/copy dataset files around
import shutil
# For parsing XML annotations
import xml.etree.ElementTree as ET
# To accept command-line inputs when running the file
import argparse

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_WIDTH  = 960
IMAGE_HEIGHT = 540
CLASS_MAP    = {"car": 0, "van": 0, "bus": 0, "others": 0}
# Consider clear vehicles and partially hidden vehicles.
INCLUDE_OCCLUSION = {0, 1}
# If FPS missing, use 25.0 as fallback
UA_DETRAC_DEFAULT_FPS = 25.0
# 7 sequences go to training
TRAIN_SEQ_RATIO = 0.70
VAL_SEQ_RATIO = 0.15
# Print progress every 100 frames
PROGRESS_EVERY = 100

# HELPER - Reads FPS from XML (Adapted for image sequences), logs where FPS comes from & falls back to 25.0 if needed.

def get_fps_from_xml(root, seq_id):

    """
    Read FPS from <sequence fps="N"> XML attribute.
    Note :Always attempt runtime extraction before using any default.

    Returns (fps_value, fps_source):
    fps_source="xml_attribute" - Read successfully from XML tag
    fps_source="default_25" - Tag absent; UA-DETRAC 25.0 used; warmup_frames will always be 10

    """
    # Reads FPS attribute from XML root
    fps_str = root.get("fps", None)
    if fps_str is not None:
        try:
            fps = float(fps_str)
            if 0 < fps <= 120:
                print(f" FPS from XML attribute : {fps}")
                return round(fps, 2), "xml_attribute"
        except ValueError:
            pass
    print(f" WARNING [{seq_id}] : No fps attribute found in XML.")
    print(f" Using UA-DETRAC default: {UA_DETRAC_DEFAULT_FPS} fps")
    print(f" fps_source logged as : 'default_25'")
    print(f" warmup_frames will be: 10 (max(10, int(0.4*25.0)))")
    return UA_DETRAC_DEFAULT_FPS, "default_25"

# HELPER - Robust frame number parser

def jpg_to_frame_num(filename):
    """
    Parses the 1-based  UA-DETRAC JPG filename.
    Handles both naming formats:
    img00001.jpg -> 1 (standard UA-DETRAC)
    000001.jpg -> 1 (rare variant without img prefix)
    Returns None if filename cannot be parsed.
    """
    name = os.path.splitext(filename)[0]  # Remove .jpg
    frame_str = name.replace("img", "")
    # If nothing left, fail safely
    if not frame_str:
        return None
    try:
        return int(frame_str)
    except ValueError:
        return None

def get_jpg_frames(seq_id, images_dir):
    import glob, os

    seq_dir = os.path.join(images_dir, seq_id)

    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(seq_dir)

    jpg_paths = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))

    if len(jpg_paths) == 0:
        raise ValueError(f"No JPG files in {seq_dir}")

    return list(enumerate(jpg_paths))


def get_warmup_frames(fps):
    return max(10, int(0.4 * fps))


# HELPER - XML parser with error counters

# Function reads one XML file and extracts bounding boxes frame by frame.
def parse_annotation_xml(xml_path, seq_id):
    """
    Parse UA-DETRAC XML file. Root tag: <sequence name = "MVI_XXXXX">
    Occlusion 2 (heavy) & 3 (full) excluded.
    Returns (root - root element for FPS extraction, frames_dict: list of bbox dicts, parse_errors : count of malformed entried that were skipped).
    """
    # Opens XML file and gets its root element.
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # A dictionary stores all frame annotations & a counter for malformed XML data
    frames_dict = {}
    parse_errors = 0

    # Loops through every <frame> tag in XML
    for frame_elem in root.findall("frame"):
        try:
            # Read the frame number
            frame_num = int(frame_elem.get("num"))
        except (TypeError, ValueError):
            parse_errors += 1
            continue
        # Prepares a list of bounding boxes for this frame
        targets = []
        # Loops through all targets in the frame
        for target in frame_elem.findall(".//target"):

            # Get the <attribute> tag. Skip if missing.
            occ_tag = target.find("occlusion")
            if occ_tag is None:
                occ = 0 # No occlusion tag - vehicle is clear
                
            else:
                # Read occlusion level
                region = occ_tag.find("region_overlap")
                if region is None:
                    occ = 0
                else:
                    status = region.get("occlusion_status", "0")
                    occ = 1 if status == "0" else 2

            if occ not in INCLUDE_OCCLUSION:
                continue

            # Vehicle type filter
            attr = target.find("attribute")
            if attr is None:
                continue
            vtype = attr.get("vehicle_type", "car")

            # Get bounding box info. If missing, count an error.
            box = target.find("box")
            if box is None:
                parse_errors += 1
                continue
            try:
                # Read box coordinates and size.
                left = float(box.get("left"))
                top = float(box.get("top"))
                width = float(box.get("width"))
                height = float(box.get("height"))
            except (TypeError, ValueError):
                parse_errors += 1
                continue
                # Store the box for this frame
            targets.append({
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "class_id": CLASS_MAP[vtype],
                "occlusion": occ,
                "vehicle_type": vtype,
            })
        # Save all boxes under that frame number
        frames_dict[frame_num] = targets
    #Return XML root, frame annotations, number of parse errors
    return root, frames_dict, parse_errors

# HELPER - Converts normal bounding box into YOLO format

def bbox_to_yolo(left, top, width, height, img_w, img_h):
    """
    Convert absolute bbox to YOLO format. All clamped to [0,1] for safety.

    """
    # Find center point of box and normalise by image size.
    cx = (left + width / 2.0) / img_w
    cy = (top + height / 2.0) / img_h
    w = width / img_w
    h = height / img_h
    # Clamp all values into the range [0,1] for safety
    return (max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)),
            max(0.0, min(1.0, w)), max(0.0, min(1.0, h)))

# MAIN - Process one sequence

# Function processes whole sequence. Inputs : sequence ID, raw image folder root, XML file path, output images folder, output labels folder
def ingest_sequence(seq_id, images_dir, xml_path, out_images_dir, out_labels_dir):

    """
    Process one UA-DETRAC sequence.
    Reads sorted JPGs from images_dir/seq_id/.
    Copies images and writes YOLO label files to output directories.
    Returns a detailed metadata dict.

    Frame accounting invariant (asserted after processing):
       frames_written + frames_skipped_warmup + invalid_jpg_count <= total_frames

    """
    # 55 : Simply aesthetically pleasing number
    print(f"\n{'='*55}")
    print(f" Processing : {seq_id}")
    print(f"{'='*55}")

    # parse XML annotations & shows how many were parsed vs problems occurred
    xml_root, annotations, parse_errors = parse_annotation_xml(xml_path, seq_id)
    print(f" Annotated frames in XML : {len(annotations)}")
    print(f" XML parse errors : {parse_errors}")

    # Extracts FPS from XML; if missing falls back to 25
    fps, fps_source = get_fps_from_xml(xml_root, seq_id)

    # Compute how many initial frames to skip
    warmup_frames = get_warmup_frames(fps)
    print(f" Warmup frames : {warmup_frames}")
    # Explains fallback behaviour
    if fps_source == "default_25":
        print(f" NOTE: fps_source=default_25 -> warmup always 10")

    # List and sort all JPG frames in the sequence folder
    seq_img_dir = os.path.join(images_dir, seq_id)
    frames_list = get_jpg_frames(seq_id, images_dir)
    all_jpgs = [os.path.basename(path) for _, path in frames_list]
    total_frames = len(all_jpgs)

    # If sample is corrupted, 960x540 is used.
    # Verify actual image dimensions from first readable frame.
    verified_width = IMAGE_WIDTH
    verified_height = IMAGE_HEIGHT
    resolution_verified = False
    # One sample per sequence - limitation documented
    for jpg_file in all_jpgs:
        sample = cv2.imread(os.path.join(seq_img_dir, jpg_file))
        if sample is not None:
            verified_height, verified_width = sample.shape[:2]
            resolution_verified = True
            print(f" Verified resolution : "
                    f"{verified_width}x{verified_height}")
            break
    if not resolution_verified:
        print(f" WARNING: Could not verify resolution. "
              f"Using fallback {IMAGE_WIDTH}x{IMAGE_HEIGHT}.")

    # Process each frame
    frames_written = 0
    frames_skipped = 0
    empty_label_count = 0
    invalid_jpg_count = 0

    for jpg_file in all_jpgs:

        # Parse frame number from filename
        frame_num = jpg_to_frame_num(jpg_file)
        if frame_num is None:
            print(f" [WARN] Cannot parse frame number: {jpg_file}")
            # Skip bad filenames
            invalid_jpg_count += 1
            continue
        # Progress counter so the terminal does not look frozen (every 100 frames)
        if frame_num % PROGRESS_EVERY == 0:
            print(f" ... frame {frame_num:05d}/{total_frames}")

        # Skip warmup frames
        # UA-DETRAC uses 1-based frame numbering.
        # Skip frame_num 1 through warmup_frames inclusive.
        if frame_num <= warmup_frames:
            frames_skipped += 1
            continue

        # Read image with error handling
        src_image_path = os.path.join(seq_img_dir, jpg_file)
        try:
            img = cv2.imread(src_image_path)
            if img is None:
                raise ValueError("cv2.imread returned None")

        except Exception as e:
            print(f" [WARN] Cannot read {jpg_file}: {e}")
            invalid_jpg_count += 1
            continue

        # Build output filename stem, eg: MVI_20011_img00015
        stem = f"{seq_id}_{os.path.splitext(jpg_file)[0]}"

        # Copy image to _all folder
        shutil.copy2(src_image_path, os.path.join(out_images_dir, stem + ".jpg"))

        # Look up bounding box for that frame, if none, return empty list.
        bboxes = annotations.get(frame_num, [])
        if not bboxes:
            empty_label_count += 1

        with open(os.path.join(out_labels_dir, stem + ".txt"), "w") as f:
            for bb in bboxes:
                # Convert from absolute pixel to normalised YOLO box
                cx, cy, w, h = bbox_to_yolo(
                    bb["left"], bb["top"],
                    bb["width"], bb["height"],
                    verified_width, verified_height)
                f.write(f"{bb['class_id']} "
                        f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        frames_written += 1

    # Frame accounting check
    # Total accounted frames must never exceed total frames found.
    accounted = frames_written + frames_skipped + invalid_jpg_count
    assert accounted <= total_frames, (
        f"[{seq_id}] Frame accounting error: "
        f"written({frames_written} + skipped({frames_skipped})) + "
        f"invalid({invalid_jpg_count}) = {accounted} "
        f"> total_frames({total_frames}). "
        f"Check for duplicate filenames or a counting bug.")

    print(f" Frame accounting   : "
          f"{frames_written}+{frames_skipped}+{invalid_jpg_count}"
          f"={accounted} <= {total_frames} ✓")

    print(f" Warmup skipped     : {frames_skipped}")
    print(f" Frames written     : {frames_written}")
    print(f" Empty label files     : {empty_label_count}")
    print(f" Invalid JPGs skipped     : {invalid_jpg_count}")

    return {
        "seq_id": seq_id,
        "fps": fps,
        "fps_source": fps_source,
        "warmup_frames": warmup_frames,
        "warmup_note": (
            "warmup=10 because fps_source=default_25"
            if fps_source == "default_25"
            else "warmup=max(10,int(0.4*fps))"),
        "total_frames": total_frames,
        "frames_written": frames_written,
        "frames_skipped_warmup": frames_skipped,
        "empty_label_files": empty_label_count,
        "invalid_jpg_count": invalid_jpg_count,
        "xml_parse_errors": parse_errors,
        "frame_accounting_ok": (accounted <= total_frames),
        "image_width": verified_width,
        "image_height": verified_height,
        "resolution_verified": resolution_verified,
        "resolution_note": (
            "Verified from first readable frame per sequence."
            "One sample only- mixed or corrupted frames later in the sequence would not be detected. Limitation acknowledged."),
        "ingestion_mode": "image_sequence",
    }

# MAIN - Sequence-level split with per-sequence frame counts
def split_by_sequence(processed_seq_ids, images_all_dir, labels_all_dir, img_train, img_val, img_test, lbl_train, lbl_val, lbl_test):
    """
    Split at sequence level to prevent temporal leakage.
    Sequences are sorted alphabetically before splitting - making the split fully deterministic without needing a random seed.

    With 10 sequences : 7 train/ 1 val/ 2 test.
    Single validation sequence is a known limitation of 10-clip constraint

    Per sequence frame counts are recorded and returned for inclusion in metadata.json for full reproducibility
    """
    sorted_seqs = sorted(processed_seq_ids)
    n = len(sorted_seqs)
    n_train = max(1, int(n * TRAIN_SEQ_RATIO))
    n_val = max(1, int(n * VAL_SEQ_RATIO))

    train_seqs = sorted_seqs[:n_train]
    val_seqs = sorted_seqs[n_train : n_train + n_val]
    test_seqs = sorted_seqs[n_train + n_val :]

    print(f"\n Sequence-level split (prevents temporal leakage):")
    print(f" Train ({len(train_seqs)}): {train_seqs}")
    print(f" Val ({len(val_seqs)}): {val_seqs}")
    print(f" Test ({len(test_seqs)}): {test_seqs}")

    # Stores which sequence belongs to which split
    split_map = {}
    for s in train_seqs: split_map[s] = "train"
    for s in val_seqs: split_map[s] = "val"
    for s in test_seqs: split_map[s] = "test"
    # Creates image directory map
    img_dirs = {"train": img_train, "val": img_val, "test": img_test}
    lbl_dirs = {"train": lbl_train, "val": lbl_val, "test": lbl_test}
    counts = {"train": 0, "val": 0, "test": 0}
    # Counts how many frames each sequence contributes
    per_seq_frame_count = {}

    all_stems = sorted([os.path.splitext(f)[0] for f in os.listdir(images_all_dir) if f.endswith(".jpg")])

    for stem in all_stems:
        # Stem format: MVI_20011_img00015
        # Extract sequence ID as first two underscore-separated parts
        parts = stem.split("_")
        seq_id = parts[0] + "_" + parts[1]
        split = split_map.get(seq_id, "test")

        shutil.copy2(
            os.path.join(images_all_dir, stem + ".jpg"),
            os.path.join(img_dirs[split], stem + ".jpg"))
        shutil.copy2(
            os.path.join(labels_all_dir, stem + ".txt"),
            os.path.join(lbl_dirs[split], stem + ".txt"))

        counts[split] += 1
        per_seq_frame_count[seq_id] = \
            per_seq_frame_count.get(seq_id, 0) + 1

    return counts, {"train": train_seqs, "val": val_seqs, "test": test_seqs}, per_seq_frame_count

# ENTRY POINT

def main(clips_master, images_dir, annotations_dir, processed_dir, logs_dir):
    """
    Actual cache files are written by detector.py on Day 5

    """
    # Setting seed control
    from src.seed_control import set_all_seeds
    set_all_seeds()

    # clips_master.txt integrity check
    # Asserts exactly 10 unique non-empty lines - no duplicates, no blank lines. Catches common file editing mistakes
    with open(clips_master, "r") as f:
        raw_lines = f.readlines()
    seq_ids = [line.strip() for line in raw_lines if line.strip()]
    # Deduplicate, preserve order
    unique_ids = list(dict.fromkeys(seq_ids))

    assert len(seq_ids) == 10, (
        f"clips_master.txt must have exactly 10 non-empty lines. "
        f"Found {len(seq_ids)}. Check for blank lines or duplicates.")

    assert len(unique_ids) == len(seq_ids), (
        f"Duplicate sequence IDs: "
        f"{[s for s in seq_ids if seq_ids.count(s) > 1]}")

    print(f"clips_master.txt check : {len(seq_ids)} unique IDs  ✓")
    print(f"Sequences to process : {seq_ids}\n")

    # Create all output directories
    images_all = os.path.join(processed_dir, "images", "_all")
    labels_all = os.path.join(processed_dir, "labels", "_all")
    img_train = os.path.join(processed_dir, "images", "train")
    img_val = os.path.join(processed_dir, "images", "val")
    img_test = os.path.join(processed_dir, "images", "test")
    lbl_train = os.path.join(processed_dir, "labels", "train")
    lbl_val = os.path.join(processed_dir, "labels", "val")
    lbl_test = os.path.join(processed_dir, "labels", "test")

    for d in [images_all, labels_all, img_train, img_val, img_test, lbl_train, lbl_val, lbl_test, logs_dir]:
        os.makedirs(d, exist_ok=True)

    # Dictionary storing metadata per sequence
    all_metadata = {}
    # Process each sequence
    processed_seq_ids = []

    for seq_id in unique_ids:
        seq_img_dir = os.path.join(images_dir, seq_id)
        xml_path = os.path.join(annotations_dir, seq_id + ".xml")

        if not os.path.isdir(seq_img_dir):
            print(f"[SKIP] Image folder not found : {seq_img_dir}")
            continue
        if not os.path.exists(xml_path):
            print(f"[SKIP] XML not found : {xml_path}")
            continue

        meta = ingest_sequence(seq_id, images_dir, xml_path, images_all, labels_all)

        all_metadata[seq_id] = meta
        processed_seq_ids.append(seq_id)

    if not all_metadata:
        print("\n[ERROR] No sequences were processed. "
              "Check that your --images_dir and --annotations_dir "
              "paths point to the correct folders.")
        return

    # Sequence-level split
    print("\n --- Sequence-level train/val/test split ---")
    split_counts, split_seqs, per_seq_frames = split_by_sequence(processed_seq_ids, images_all, labels_all, img_train, img_val, img_test, lbl_train, lbl_val, lbl_test)

    # Write full metadata.json
    metadata_path = os.path.join(logs_dir, "metadata.json")
    full_output = {
        "ingestion_mode": "image_sequence",
        "train_count": split_counts["train"],
        "val_count": split_counts["val"],
        "test_count": split_counts["test"],
        "split_seqs": split_seqs,
        "per_seq_frame_counts": per_seq_frames,
        "fps_note": (
            "fps_source='xml_attribute': read from XML <sequence fps=> tag. "
            "fps_source='default_25': tag absent; UA-DETRAC standard "
            "25.0 fps used; warmup_frames=10 in all such cases."),
        "split_note": (
            "Sequence-level split 7/1/2. Constrained engineering split for 10 clips. Not benchmark-optimal. Single validation sequence is a known limitation acknowledged."
            "Resolution verified from one sample frame per sequence."),
        "partition_verification_note": (
            "Range-based verification: MVI_20xxx and MVI_40xxx are official UA-DETRAC training ranges. Machine-generated evidence in logs/day2_train_partition_verification.json."),
        "sequences": all_metadata,
    }
    with open(metadata_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n metadata.json -> {metadata_path}")

    # Short summary JSON
    summary_path = os.path.join(logs_dir, "day2_summary.json")
    summary = {
        "day": 2,
        "sequences_processed": list(all_metadata.keys()),
        "total_sequences": split_counts["train"],
        "val_count": split_counts["val"],
        "split_seqs": split_seqs,
        "per_seq_frame_counts": per_seq_frames,
        "fps_per_seq": {
            sid: {"fps": m["fps"], "fps_source": m["fps_source"]}
            for sid, m in all_metadata.items()},
        "ingestion_mode": "image_sequence"
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f" day2_summary.json -> {summary_path}")

    # Print final summary to terminal
    print("\n" + "="*55)
    print(" Day 2 - Final Summary")
    print("="*55)
    for sid, m in all_metadata.items():
        print(f" {sid}: fps={m['fps']} ({m['fps_source']}), "
              f"written={m['frames_written']}, "
              f"empty={m['empty_label_files']}, "
              f"invalid={m['invalid_jpg_count']}, "
              f"accounting={'✓' if m['frame_accounting_ok'] else '✗'}")
    print(f"\n Train: {split_counts['train']} frames "
          f"- seqs: {split_seqs['train']}")
    print(f" Val: {split_counts['val']} frames "
          f"- seqs: {split_seqs['val']}")
    print(f" Test: {split_counts['test']} frames "
          f"- seqs: {split_seqs['test']}")
    print("="*55)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "UA-DETRAC Ingestor - Day 2. "
            "Run as: python -m src.ingestor_uadetrac "
            "--clips_master configs/clips_master.txt ..."))

    parser.add_argument("--clips_master", required=True, help="Root folder containing MVI_XXXXX subfolders")
    parser.add_argument("--images_dir", required=True, help="Root folder containing MVI_XXXX subfolders")
    parser.add_argument("--annotations_dir", required=True, help="Folder containing MVI_XXXXX.xml files")
    parser.add_argument("--processed_dir", required=True, help="Output processed data folder")
    parser.add_argument("--logs_dir", required=True, help="Output logs folder for metadata.json")
    args = parser.parse_args()
    main(args.clips_master, args.images_dir, args.annotations_dir, args.processed_dir, args.logs_dir)