import os, json

def write_cache(seq_id, split, detector_name, frame_results, warmup_boundary, cache_dir):
    cache_dir = os.path.join(cache_dir, split, detector_name)
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{seq_id}.json")

    if os.path.exists(cache_path):
        raise FileExistsError(f"Cache already exists: {cache_path}")

    output = {
        "seq_id": seq_id,
        "split": split,
        "detector": detector_name,
        "total_frames": len(frame_results),
        "warmup_boundary": warmup_boundary,
        "frames": frame_results
    }

    with open(cache_path, "w") as f:
        json.dump(output, f)

    return cache_path


def load_cache(seq_id, split, detector_name, cache_dir):
    cache_path = os.path.join(cache_dir, split, detector_name, f"{seq_id}.json")

    if not os.path.exists(cache_path):
        raise FileNotFoundError(cache_path)

    with open(cache_path) as f:
        data = json.load(f)

    return data["warmup_boundary"], data["frames"]