## Block B - Tracker Comparison Results

| Tracker | MOTA | IDF1 | IDSW | Frag | FPS (median) | Selected |
|---|---|---|---|---|---|---|
| SORT | -0.7775 | 0.0412 | 31 | 31 | 1129.65 | YES |
| DeepSORT (VeRi-776) | -1.3700 | 0.0008 | 0 | 0 | 1063.68 | - |
| BYTETRACK | -2.0720 | 0.0006 | 0 | 0 | 1231.37 | - |

**Selected tracker: SORT**
**Rationale:** Selected 'sort': idf1_mean=0.0412, idsw_total=31, fps_median=1129.65. Tracker frozen for Blocks C and D.

> **Statistical note (Pearson r):** Computed on n=2 test sequences -- r is degenerate (always +/-1.0 for 2 data points). This is a methodological scaffold; Block C will provide n=10 for a meaningful result.

---

## Block B - Post-Processing Impact

| Tracker | Tracks Before | Tracks After | Discarded | Frames Interpolated |
|---|---|---|---|---|
| SORT | 112 | 76 | 36 | 246 |
| DeepSORT (VeRi-776) | 134 | 89 | 45 | 41 |
| BYTETRACK | 658 | 370 | 288 | 73 |

*(Values summed across MVI_20062 and MVI_20063)*

Fragment filter: `min_track_length=15` (Fix F11).
Interpolation: `max_interp_gap=3 frames` (Fix F22). Gaps >3 frames: track split. No ghost interpolation.
Block C reads exclusively from `_final.json` files.
