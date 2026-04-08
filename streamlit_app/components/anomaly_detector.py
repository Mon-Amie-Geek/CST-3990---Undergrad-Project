"""
anomaly_detector.py

Anomaly detection module implementing rule-based, One-Class SVM, and Isolation Forest methods.
OC-SVM nu=0.01 aligns with Block C Day 12 best-nu result (F2_only, AUROC=0.9935).
If the pre-trained OC-SVM (logs/block_c/ocsvm_trained_best.pkl) and MinMaxScaler
(models/minmax_scaler.pkl) are available, the OC-SVM option loads them directly
instead of training on warmup data — this gives Block C research-quality inference.
Student: MANJOO Ameera Najla | M01014463
"""

import os
import numpy as np
from collections import deque
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

# ── Paths to pre-trained artefacts (Block C) ─────────────────────────────────
_COMP_DIR    = os.path.dirname(__file__)                         # streamlit_app/components/
_OCSVM_PKL   = os.path.normpath(os.path.join(
    _COMP_DIR, "..", "..", "logs", "block_c", "ocsvm_trained_best.pkl"))
_SCALER_PKL  = os.path.normpath(os.path.join(
    _COMP_DIR, "..", "..", "models", "minmax_scaler.pkl"))

# ── Block C OC-SVM feature order (F2_only): vel_px_sec, vel_px_sec_smooth ───
# The scaler was fitted on 7 columns in this order (SCALER_FIT_COLUMNS):
#   [vel_px_sec, vel_px_sec_smooth, vehicle_count, roi_occupancy,
#    inter_vehicle_dist_norm, dwell_time_sec, proximity_count_rolling]
# For F2-only inference we build a 7-column row and pass only the two speed columns
# to the loaded OC-SVM (which was fitted on 2-column input after scaler transform).
_SCALER_COLS = [
    "vel_px_sec", "vel_px_sec_smooth",
    "vehicle_count", "roi_occupancy",
    "inter_vehicle_dist_norm", "dwell_time_sec",
    "proximity_count_rolling",
]
_F2_ONLY_IDXS = [0, 1]  # positions of vel_px_sec, vel_px_sec_smooth in SCALER_COLS


def _try_load_pretrained():
    """
    Attempt to load the Block C pre-trained OC-SVM and MinMaxScaler.
    Returns (ocsvm, scaler) or (None, None) if files are not available.
    """
    try:
        import joblib
        if os.path.exists(_OCSVM_PKL) and os.path.exists(_SCALER_PKL):
            ocsvm  = joblib.load(_OCSVM_PKL)
            scaler = joblib.load(_SCALER_PKL)
            return ocsvm, scaler
    except Exception:
        pass
    return None, None


class AnomalyDetector:
    """
    Detect anomalous vehicle behaviours using configurable methods.

    OC-SVM path:
      - If the Block C pre-trained model (ocsvm_trained_best.pkl) and scaler
        (minmax_scaler.pkl) are found at repo root, they are loaded directly.
        Inference uses [mean_speed_px_sec, mean_speed_px_sec_smooth] as F2_only features.
      - Otherwise, the model is trained on warmup data (first `warmup_frames` frames)
        using nu=0.01 (Block C Day 12 best nu).

    Isolation Forest: always trained on warmup data (no pre-trained artefact).
    Rule-Based: online Z-score (mean ± 2×std) from rolling history.
    """

    def __init__(self, method="rule_based"):
        """
        Initialise anomaly detector.

        Args:
            method (str): One of "rule_based", "ocsvm", "isolation_forest"
        """
        self.method        = method
        self.speed_history = deque(maxlen=500)
        self.dist_history  = deque(maxlen=500)
        self.warmup_frames = 200      # collect normal stats before predicting
        self.frame_count   = 0

        # Rule-based statistics
        self.speed_mean = None
        self.speed_std  = None
        self.dist_mean  = None
        self.dist_std   = None

        # Rolling smoothed speed (approximate vel_px_sec_smooth for OC-SVM)
        self._speed_buf     = deque(maxlen=5)   # 5-frame window ≈ 0.2s at 25 FPS

        # Feature history for warmup training
        self.feature_history = []

        # ML models
        self.ocsvm            = None
        self.isolation_forest = None
        self.is_trained       = False
        self._pretrained_ocsvm   = None
        self._pretrained_scaler  = None
        self._using_pretrained   = False

        # Try to load Block C pre-trained artefacts for OC-SVM
        if self.method == "ocsvm":
            _m, _s = _try_load_pretrained()
            if _m is not None:
                self._pretrained_ocsvm  = _m
                self._pretrained_scaler = _s
                self._using_pretrained  = True

    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, feature_dict):
        """
        Predict if the current frame contains anomalies.

        Args:
            feature_dict (dict): Output of FeatureExtractor.extract(), containing at minimum:
                - mean_speed_px_sec  (float)
                - min_distance_norm  (float)
                - mean_dwell_sec     (float)  [optional]
                - vehicle_count      (int)    [optional]
                - roi_occupancy      (float)  [optional]

        Returns:
            dict: {is_anomaly, score, method, triggered_feature}
        """
        self.frame_count += 1

        speed    = float(feature_dict.get("mean_speed_px_sec", 0.0))
        distance = float(feature_dict.get("min_distance_norm", 1.0))
        dwell    = float(feature_dict.get("mean_dwell_sec", 0.0))
        v_count  = float(feature_dict.get("vehicle_count", 0))
        roi_occ  = float(feature_dict.get("roi_occupancy", 0.0))

        # Maintain rolling smoothed speed (approx. vel_px_sec_smooth)
        self._speed_buf.append(speed)
        speed_smooth = float(np.mean(self._speed_buf))

        self.speed_history.append(speed)
        self.dist_history.append(distance)

        _no_anomaly = {
            "is_anomaly": False, "score": 0.0,
            "method": self.method, "triggered_feature": None
        }

        # Warmup: collect data, no predictions
        if self.frame_count <= self.warmup_frames:
            self.feature_history.append([speed, speed_smooth])
            return _no_anomaly

        # Train warmup-based models once (frame warmup_frames + 1)
        if not self.is_trained:
            self._train()

        # Pre-trained OC-SVM: skip warmup training, use Block C model directly
        if self.method == "ocsvm" and self._using_pretrained:
            return self._predict_ocsvm_pretrained(speed, speed_smooth, v_count, roi_occ, distance, dwell)

        if self.method == "rule_based":
            return self._predict_rule_based(speed, distance)
        elif self.method == "ocsvm":
            return self._predict_ocsvm_warmup(speed, speed_smooth)
        elif self.method == "isolation_forest":
            return self._predict_isolation_forest(speed, distance)

        return _no_anomaly

    # ─────────────────────────────────────────────────────────────────────────

    def _train(self):
        """Train ML-based detectors on warmup data (used when pre-trained not available)."""
        if len(self.feature_history) < 10:
            self.is_trained = True
            return

        X = np.array(self.feature_history)  # shape (N, 2): [speed, speed_smooth]

        if self.method == "ocsvm":
            # nu=0.01: Block C Day 12 best nu for F2_only (AUROC=0.9935)
            self.ocsvm = OneClassSVM(kernel="rbf", nu=0.01)
            try:
                self.ocsvm.fit(X)
            except Exception as e:
                print(f"[AnomalyDetector] OC-SVM warmup training failed: {e}")

        elif self.method == "isolation_forest":
            self.isolation_forest = IsolationForest(
                n_estimators=100, contamination=0.05, random_state=42
            )
            try:
                self.isolation_forest.fit(X)
            except Exception as e:
                print(f"[AnomalyDetector] IsolationForest warmup training failed: {e}")

        self.is_trained = True

    # ─────────────────────────────────────────────────────────────────────────

    def _predict_rule_based(self, speed, distance):
        """Rule-based anomaly detection using Z-score thresholds (mean ± 2×std)."""
        if len(self.speed_history) < 10:
            return {"is_anomaly": False, "score": 0.0,
                    "method": self.method, "triggered_feature": None}

        speed_arr = np.array(list(self.speed_history))
        dist_arr  = np.array(list(self.dist_history))

        s_mean, s_std = float(np.mean(speed_arr)), float(np.std(speed_arr))
        d_mean, d_std = float(np.mean(dist_arr)),  float(np.std(dist_arr))

        speed_high = s_mean + 2.0 * s_std
        dist_low   = d_mean - 2.0 * d_std

        is_anomaly      = False
        triggered_feat  = None

        if speed > speed_high:
            is_anomaly     = True
            triggered_feat = f"High Speed ({speed:.1f} > {speed_high:.1f} px/s)"

        if distance < dist_low and not is_anomaly:
            is_anomaly     = True
            triggered_feat = f"Low Distance ({distance:.3f} < {dist_low:.3f})"

        score = (speed - s_mean) / (s_std + 1e-6)

        return {
            "is_anomaly":        is_anomaly,
            "score":             float(abs(score)),
            "method":            self.method,
            "triggered_feature": triggered_feat,
        }

    def _predict_ocsvm_warmup(self, speed, speed_smooth):
        """OC-SVM trained on warmup data (fallback when pre-trained not available)."""
        if self.ocsvm is None:
            return {"is_anomaly": False, "score": 0.0,
                    "method": self.method, "triggered_feature": None}
        try:
            X    = np.array([[speed, speed_smooth]])
            pred = self.ocsvm.predict(X)[0]
            scr  = -float(self.ocsvm.decision_function(X)[0])
            return {
                "is_anomaly":        pred == -1,
                "score":             abs(scr),
                "method":            self.method,
                "triggered_feature": "OC-SVM (speed)" if pred == -1 else None,
            }
        except Exception as e:
            print(f"[AnomalyDetector] OC-SVM warmup prediction failed: {e}")
            return {"is_anomaly": False, "score": 0.0,
                    "method": self.method, "triggered_feature": None}

    def _predict_ocsvm_pretrained(self, speed, speed_smooth, v_count, roi_occ, distance, dwell):
        """
        OC-SVM inference using the Block C pre-trained model (F2_only, nu=0.01).
        Builds the full 7-column scaled row, extracts F2 columns, and queries the model.
        """
        try:
            # Build full 7-column vector (fill unavailable F3 with 0 — NaN-safe approximation)
            x_full = np.array([[
                speed, speed_smooth, v_count, roi_occ,
                distance, dwell, 0.0  # proximity_count_rolling not available in streaming
            ]])
            # Scale with the fitted MinMaxScaler
            x_scaled = self._pretrained_scaler.transform(x_full)
            # Extract F2-only columns (indices 0,1: vel_px_sec, vel_px_sec_smooth)
            x_f2 = x_scaled[:, _F2_ONLY_IDXS]
            pred = self._pretrained_ocsvm.predict(x_f2)[0]
            scr  = -float(self._pretrained_ocsvm.decision_function(x_f2)[0])
            return {
                "is_anomaly":        pred == -1,
                "score":             abs(scr),
                "method":            "ocsvm (Block C F2_only)",
                "triggered_feature": "OC-SVM [Block C F2_only]" if pred == -1 else None,
            }
        except Exception as e:
            print(f"[AnomalyDetector] Pre-trained OC-SVM prediction failed: {e}")
            # Fallback to warmup-based if loaded but fails
            return self._predict_ocsvm_warmup(speed, speed_smooth)

    def _predict_isolation_forest(self, speed, distance):
        """Isolation Forest anomaly detection."""
        if self.isolation_forest is None:
            return {"is_anomaly": False, "score": 0.0,
                    "method": self.method, "triggered_feature": None}
        try:
            X    = np.array([[speed, distance]])
            pred = self.isolation_forest.predict(X)[0]
            scr  = -float(self.isolation_forest.score_samples(X)[0])
            return {
                "is_anomaly":        pred == -1,
                "score":             abs(scr),
                "method":            self.method,
                "triggered_feature": "Isolation Forest" if pred == -1 else None,
            }
        except Exception as e:
            print(f"[AnomalyDetector] IsolationForest prediction failed: {e}")
            return {"is_anomaly": False, "score": 0.0,
                    "method": self.method, "triggered_feature": None}

    # ─────────────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset detector state for a new video."""
        self.speed_history.clear()
        self.dist_history.clear()
        self._speed_buf.clear()
        self.feature_history  = []
        self.frame_count      = 0
        self.is_trained       = False
        self.ocsvm            = None
        self.isolation_forest = None
        # Pre-trained artefacts are NOT reset — they persist across videos
