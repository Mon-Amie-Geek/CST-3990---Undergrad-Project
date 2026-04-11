"""
anomaly_detector.py

Anomaly detection module implementing rule-based, One-Class SVM, and
Isolation Forest methods.
OC-SVM nu=0.01 aligns with Block C Day 12 best-nu result (F2_only, AUROC=0.9935).
If the pre-trained OC-SVM (logs/block_c/ocsvm_trained_best.pkl) and MinMaxScaler
(models/minmax_scaler.pkl) are available, the OC-SVM option loads them directly
instead of training on warmup data when the selected feature set is exactly
F2_only. Other feature-set choices are trained online from warmup data so every
Block C and Block D combination works coherently.
Student: MANJOO Ameera Najla | M01014463
"""

import logging
import os
import warnings
from collections import deque

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Paths to pre-trained artefacts (Block C)
_COMP_DIR = os.path.dirname(__file__)
_OCSVM_PKL = os.path.normpath(
    os.path.join(_COMP_DIR, "..", "..", "logs", "block_c", "ocsvm_trained_best.pkl")
)
_SCALER_PKL = os.path.normpath(
    os.path.join(_COMP_DIR, "..", "..", "models", "minmax_scaler.pkl")
)

# Block C OC-SVM feature order (F2_only)
_SCALER_COLS = [
    "vel_px_sec",
    "vel_px_sec_smooth",
    "vehicle_count",
    "roi_occupancy",
    "inter_vehicle_dist_norm",
    "dwell_time_sec",
    "proximity_count_rolling",
]
_F2_ONLY_IDXS = [0, 1]
_LOGGER = logging.getLogger(__name__)


def _try_load_pretrained():
    """
    Attempt to load the Block C pre-trained OC-SVM and MinMaxScaler.
    Returns (ocsvm, scaler) or (None, None) if files are not available.
    """
    try:
        import joblib

        if not (os.path.exists(_OCSVM_PKL) and os.path.exists(_SCALER_PKL)):
            return None, None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = joblib.load(_OCSVM_PKL)
            scaler = joblib.load(_SCALER_PKL)

        ocsvm = raw.get("model") if isinstance(raw, dict) else raw
        if ocsvm is None or not hasattr(ocsvm, "predict"):
            return None, None
        return ocsvm, scaler
    except Exception:
        return None, None


class AnomalyDetector:
    """
    Detect anomalous vehicle behaviours using configurable methods.

    OC-SVM path:
      - When the selected feature set is exactly F2_only and the Block C
        artefacts exist, the pre-trained model is used directly.
      - Otherwise the model is trained on warmup data collected from the
        selected live feature groups.

    Isolation Forest:
      - Always trained on warmup data collected from the selected groups.

    Rule-Based:
      - Applies simple rolling thresholds using whichever feature groups are
        currently selected.
    """

    def __init__(self, method="rule_based"):
        self.method = method
        self.speed_history = deque(maxlen=500)
        self.dist_history = deque(maxlen=500)
        self.count_history = deque(maxlen=500)
        self.warmup_frames = 200
        self.frame_count = 0

        # Rolling smoothed speed (approximate vel_px_sec_smooth)
        self._speed_buf = deque(maxlen=5)

        # Feature history for warmup training
        self.feature_history = []

        # ML models
        self.ocsvm = None
        self.isolation_forest = None
        self.is_trained = False
        self._pretrained_ocsvm = None
        self._pretrained_scaler = None
        self._using_pretrained = False

        if self.method == "ocsvm":
            model, scaler = _try_load_pretrained()
            if model is not None:
                self._pretrained_ocsvm = model
                self._pretrained_scaler = scaler
                self._using_pretrained = True

    def predict(self, feature_dict):
        """
        Predict if the current frame contains anomalies.

        Args:
            feature_dict: Output of FeatureExtractor.extract()

        Returns:
            dict: {is_anomaly, score, method, triggered_feature}
        """
        self.frame_count += 1

        speed = float(feature_dict.get("mean_speed_px_sec", 0.0))
        distance = float(feature_dict.get("min_distance_norm", 1.0))
        dwell = float(feature_dict.get("mean_dwell_sec", 0.0))
        vehicle_count = float(feature_dict.get("vehicle_count", 0.0))
        roi_occupancy = float(feature_dict.get("roi_occupancy", 0.0))
        selected_features = set(feature_dict.get("selected_features") or [])

        self._speed_buf.append(speed)
        speed_smooth = float(np.mean(self._speed_buf))

        self.speed_history.append(speed)
        self.dist_history.append(distance)
        self.count_history.append(vehicle_count)

        ml_vector = self._build_ml_vector(
            speed,
            speed_smooth,
            distance,
            dwell,
            vehicle_count,
            roi_occupancy,
            selected_features,
        )
        use_pretrained_ocsvm = (
            self.method == "ocsvm"
            and self._using_pretrained
            and selected_features == {"F2: Speed"}
        )

        no_anomaly = {
            "is_anomaly": False,
            "score": 0.0,
            "method": self.method,
            "triggered_feature": None,
        }

        if self.frame_count <= self.warmup_frames:
            self.feature_history.append(ml_vector)
            return no_anomaly

        if not self.is_trained:
            if use_pretrained_ocsvm:
                self.is_trained = True
            else:
                self._train()

        if use_pretrained_ocsvm:
            return self._predict_ocsvm_pretrained(
                speed,
                speed_smooth,
                vehicle_count,
                roi_occupancy,
                distance,
                dwell,
            )

        if self.method == "rule_based":
            return self._predict_rule_based(
                speed, distance, vehicle_count, selected_features
            )
        if self.method == "ocsvm":
            return self._predict_ocsvm_warmup(ml_vector, selected_features)
        if self.method == "isolation_forest":
            return self._predict_isolation_forest(ml_vector, selected_features)

        return no_anomaly

    def _build_ml_vector(
        self,
        speed,
        speed_smooth,
        distance,
        dwell,
        vehicle_count,
        roi_occupancy,
        selected_features,
    ):
        """Build the model input vector from the selected feature groups."""
        vector = []

        if "F1: Density/Flow" in selected_features:
            vector.extend([vehicle_count, roi_occupancy])
        if "F2: Speed" in selected_features:
            vector.extend([speed, speed_smooth])
        if "F3: Distance/Proximity" in selected_features:
            vector.extend([distance, dwell])

        if not vector:
            vector = [speed, speed_smooth]

        return vector

    def _train(self):
        """Train warmup-based models on the collected live feature vectors."""
        if len(self.feature_history) < 10:
            self.is_trained = True
            return

        X = np.array(self.feature_history, dtype=float)

        if self.method == "ocsvm":
            self.ocsvm = OneClassSVM(kernel="rbf", nu=0.01)
            try:
                self.ocsvm.fit(X)
            except Exception as exc:
                _LOGGER.warning("OC-SVM warmup training failed: %s", exc)

        elif self.method == "isolation_forest":
            self.isolation_forest = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42,
            )
            try:
                self.isolation_forest.fit(X)
            except Exception as exc:
                _LOGGER.warning("IsolationForest warmup training failed: %s", exc)

        self.is_trained = True

    def _predict_rule_based(self, speed, distance, vehicle_count, selected_features):
        """Rule-based anomaly detection over the currently selected feature groups."""
        if len(self.speed_history) < 10:
            return {
                "is_anomaly": False,
                "score": 0.0,
                "method": self.method,
                "triggered_feature": None,
            }

        speed_arr = np.array(list(self.speed_history), dtype=float)
        dist_arr = np.array(list(self.dist_history), dtype=float)
        count_arr = np.array(list(self.count_history), dtype=float)

        s_mean, s_std = float(np.mean(speed_arr)), float(np.std(speed_arr))
        d_mean, d_std = float(np.mean(dist_arr)), float(np.std(dist_arr))
        c_mean, c_std = float(np.mean(count_arr)), float(np.std(count_arr))

        speed_high = s_mean + 2.0 * s_std
        dist_low = d_mean - 2.0 * d_std
        count_high = c_mean + 2.0 * c_std

        is_anomaly = False
        triggered_feature = None
        scores = []

        if "F2: Speed" in selected_features:
            scores.append(abs((speed - s_mean) / (s_std + 1e-6)))
            if speed > speed_high:
                is_anomaly = True
                triggered_feature = f"High Speed ({speed:.1f} > {speed_high:.1f} px/s)"

        if (
            "F3: Distance/Proximity" in selected_features
            and distance < dist_low
            and not is_anomaly
        ):
            scores.append(abs((d_mean - distance) / (d_std + 1e-6)))
            is_anomaly = True
            triggered_feature = f"Low Distance ({distance:.3f} < {dist_low:.3f})"
        elif "F3: Distance/Proximity" in selected_features:
            scores.append(abs((d_mean - distance) / (d_std + 1e-6)))

        if (
            "F1: Density/Flow" in selected_features
            and vehicle_count > count_high
            and not is_anomaly
        ):
            scores.append(abs((vehicle_count - c_mean) / (c_std + 1e-6)))
            is_anomaly = True
            triggered_feature = (
                f"High Density ({vehicle_count:.1f} > {count_high:.1f} vehicles)"
            )
        elif "F1: Density/Flow" in selected_features:
            scores.append(abs((vehicle_count - c_mean) / (c_std + 1e-6)))

        return {
            "is_anomaly": is_anomaly,
            "score": float(max(scores) if scores else 0.0),
            "method": self.method,
            "triggered_feature": triggered_feature,
        }

    def _predict_ocsvm_warmup(self, feature_vector, selected_features):
        """Warmup-trained OC-SVM prediction using the selected feature groups."""
        if self.ocsvm is None:
            return {
                "is_anomaly": False,
                "score": 0.0,
                "method": self.method,
                "triggered_feature": None,
            }

        try:
            X = np.array([feature_vector], dtype=float)
            pred = self.ocsvm.predict(X)[0]
            score = -float(self.ocsvm.decision_function(X)[0])
            feature_label = " + ".join(sorted(selected_features)) or "selected features"
            return {
                "is_anomaly": pred == -1,
                "score": abs(score),
                "method": self.method,
                "triggered_feature": f"OC-SVM ({feature_label})" if pred == -1 else None,
            }
        except Exception as exc:
            _LOGGER.warning("OC-SVM warmup prediction failed: %s", exc)
            return {
                "is_anomaly": False,
                "score": 0.0,
                "method": self.method,
                "triggered_feature": None,
            }

    def _predict_ocsvm_pretrained(
        self, speed, speed_smooth, vehicle_count, roi_occupancy, distance, dwell
    ):
        """
        Pre-trained Block C OC-SVM inference for the exact F2_only configuration.
        """
        try:
            x_full = pd.DataFrame(
                [[
                    speed,
                    speed_smooth,
                    vehicle_count,
                    roi_occupancy,
                    distance,
                    dwell,
                    0.0,
                ]],
                columns=_SCALER_COLS,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_scaled = self._pretrained_scaler.transform(x_full)
            x_f2 = x_scaled[:, _F2_ONLY_IDXS]
            pred = self._pretrained_ocsvm.predict(x_f2)[0]
            score = -float(self._pretrained_ocsvm.decision_function(x_f2)[0])
            return {
                "is_anomaly": pred == -1,
                "score": abs(score),
                "method": "ocsvm (Block C F2_only)",
                "triggered_feature": "OC-SVM [Block C F2_only]" if pred == -1 else None,
            }
        except Exception as exc:
            _LOGGER.warning("Pre-trained OC-SVM prediction failed: %s", exc)
            return self._predict_ocsvm_warmup([speed, speed_smooth], {"F2: Speed"})

    def _predict_isolation_forest(self, feature_vector, selected_features):
        """Warmup-trained Isolation Forest prediction over selected feature groups."""
        if self.isolation_forest is None:
            return {
                "is_anomaly": False,
                "score": 0.0,
                "method": self.method,
                "triggered_feature": None,
            }

        try:
            X = np.array([feature_vector], dtype=float)
            pred = self.isolation_forest.predict(X)[0]
            score = -float(self.isolation_forest.score_samples(X)[0])
            feature_label = " + ".join(sorted(selected_features)) or "selected features"
            return {
                "is_anomaly": pred == -1,
                "score": abs(score),
                "method": self.method,
                "triggered_feature": (
                    f"Isolation Forest ({feature_label})" if pred == -1 else None
                ),
            }
        except Exception as exc:
            _LOGGER.warning("IsolationForest prediction failed: %s", exc)
            return {
                "is_anomaly": False,
                "score": 0.0,
                "method": self.method,
                "triggered_feature": None,
            }

    def reset(self):
        """Reset detector state for a new video."""
        self.speed_history.clear()
        self.dist_history.clear()
        self.count_history.clear()
        self._speed_buf.clear()
        self.feature_history = []
        self.frame_count = 0
        self.is_trained = False
        self.ocsvm = None
        self.isolation_forest = None
