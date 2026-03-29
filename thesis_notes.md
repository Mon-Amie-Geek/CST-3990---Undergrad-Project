# Thesis Notes — Day 19 Preparation
# Compiled: Day 4
# Referenced throughout implementation days

---

## §3.8.2 — Construct Validity: Velocity and Distance Proxies

**Exact paragraph for thesis (§3.8.2):**

Velocity is computed as image-plane displacement per second (px/sec),
not world-space velocity. A vehicle moving 10 pixels near the camera
is physically slower than a vehicle moving 10 pixels in the far
background at the same pixel rate. All velocity and distance values
in this study are therefore relative behavioural indicators rather
than absolute physical measurements. This is referred to throughout
as Image-Plane Velocity to distinguish it clearly from
calibrated world-space speed estimation, which would require camera
intrinsic parameters and homography projection — outside the scope
of this comparative study.

Inter-vehicle distances are normalised by frame diagonal and
interpreted comparatively across pipeline configurations, not as
absolute metric distances. This constitutes a construct validity
limitation. The perspective disclaimer is attached to every
proximity result in logs, result tables, and figure captions
throughout Chapter 4.

---

## §3.4 — Implementation Constraint: Video Path Resolution

`resolve_video_path()` builds paths as:
DATASET_ROOT / subfolder / clip_id.[mp4|avi|mov]

This assumes clips are directly under the subfolder (flat layout).
If clips are nested one level deeper the path will not resolve.
This is an implementation constraint arising from the diversity
of dataset packaging conventions and is documented here as a
known limitation of the path resolution strategy.

---

## Warmup Integration Note (Day 10)

warmup_frames is computed and exported by Day 4
(normalisation_contexts.json).
Enforcement inside the feature extraction loop belongs in Day 10.

Thesis phrasing:
"The warmup duration is computed per-clip relative to actual FPS
and exported by the normalisation module. Enforcement — discarding
the first warmup_frames rows of features.csv — is implemented in
Day 10's generator-based feature extraction loop"

---

## Sanity Check Framing (§3.8)

The Day 4 sanity check validates formula correctness
using actual video frames and contour-based motion detection.
It is an engineering debug check, not a formal benchmark.
The centroid extraction method (largest contour + 30% bbox shift)
is heuristic and may capture background motion or illumination
change rather than true vehicle motion. 

Present as:
"an engineering sanity check confirming the FPS-relative velocity
formula produces consistent results across datasets, not as
evidence of model performance."