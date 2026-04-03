"""
anomaly_detector.py

Anomaly detection module implementing rule-based, One-Class SVM, and Isolation Forest methods.
Student: MANJOO Ameera Najla | M01014463
"""

import numpy as np
from collections import deque
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """
    Detect anomalous vehicle behaviors using configurable methods.
    """
    
    def __init__(self, method="rule_based"):
        """
        Initialize anomaly detector.
        
        Args:
            method (str): One of "rule_based", "ocsvm", "isolation_forest"
        """
        self.method = method
        self.speed_history = deque(maxlen=500)
        self.distance_history = deque(maxlen=500)
        self.warmup_frames = 200
        self.frame_count = 0
        
        # Statistical thresholds for rule-based
        self.speed_mean = None
        self.speed_std = None
        self.distance_mean = None
        self.distance_std = None
        
        # ML models
        self.ocsvm = None
        self.isolation_forest = None
        self.is_trained = False
        self.feature_history = []
    
    def predict(self, feature_dict):
        """
        Predict if current frame contains anomalies.
        
        Args:
            feature_dict (dict): {
                "mean_speed_px_sec": float,
                "min_distance_norm": float,
                ...
            }
            
        Returns:
            dict: {
                "is_anomaly": bool,
                "score": float,
                "method": str,
                "triggered_feature": str or None
            }
        """
        self.frame_count += 1
        
        # Update histories
        speed = feature_dict.get("mean_speed_px_sec", 0.0)
        distance = feature_dict.get("min_distance_norm", 1.0)
        
        self.speed_history.append(speed)
        self.distance_history.append(distance)
        
        # Warmup period: just collect data
        if self.frame_count <= self.warmup_frames:
            self.feature_history.append([speed, distance])
            return {
                "is_anomaly": False,
                "score": 0.0,
                "method": self.method,
                "triggered_feature": None
            }
        
        # Train on warmup data once
        if not self.is_trained and self.frame_count == self.warmup_frames + 1:
            self._train()
        
        # Apply detection method
        if self.method == "rule_based":
            return self._predict_rule_based(speed, distance)
        elif self.method == "ocsvm":
            return self._predict_ocsvm(speed, distance)
        elif self.method == "isolation_forest":
            return self._predict_isolation_forest(speed, distance)
        else:
            return {
                "is_anomaly": False,
                "score": 0.0,
                "method": self.method,
                "triggered_feature": None
            }
    
    def _train(self):
        """Train ML-based detectors on warmup data."""
        if len(self.feature_history) < 10:
            return
        
        X = np.array(self.feature_history)
        
        if self.method == "ocsvm":
            self.ocsvm = OneClassSVM(kernel="rbf", nu=0.05)
            try:
                self.ocsvm.fit(X)
                self.is_trained = True
            except Exception as e:
                print(f"OneClassSVM training failed: {e}")
        
        elif self.method == "isolation_forest":
            self.isolation_forest = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42
            )
            try:
                self.isolation_forest.fit(X)
                self.is_trained = True
            except Exception as e:
                print(f"IsolationForest training failed: {e}")
    
    def _predict_rule_based(self, speed, distance):
        """Rule-based anomaly detection using Z-score thresholds."""
        if len(self.speed_history) < 10:
            return {
                "is_anomaly": False,
                "score": 0.0,
                "method": self.method,
                "triggered_feature": None
            }
        
        speed_arr = np.array(list(self.speed_history))
        distance_arr = np.array(list(self.distance_history))
        
        self.speed_mean = np.mean(speed_arr)
        self.speed_std = np.std(speed_arr)
        self.distance_mean = np.mean(distance_arr)
        self.distance_std = np.std(distance_arr)
        
        # Thresholds: mean ± 2*std
        speed_threshold_high = self.speed_mean + 2 * self.speed_std
        distance_threshold_low = self.distance_mean - 2 * self.distance_std
        
        is_anomaly = False
        triggered_feature = None
        
        if speed > speed_threshold_high:
            is_anomaly = True
            triggered_feature = "High Speed"
        
        if distance < distance_threshold_low:
            is_anomaly = True
            triggered_feature = "Low Distance"
        
        score = (speed - self.speed_mean) / (self.speed_std + 1e-6)
        
        return {
            "is_anomaly": is_anomaly,
            "score": float(abs(score)),
            "method": self.method,
            "triggered_feature": triggered_feature
        }
    
    def _predict_ocsvm(self, speed, distance):
        """One-Class SVM anomaly detection."""
        if self.ocsvm is None:
            return {
                "is_anomaly": False,
                "score": 0.0,
                "method": self.method,
                "triggered_feature": None
            }
        
        try:
            X = np.array([[speed, distance]])
            prediction = self.ocsvm.predict(X)[0]
            score = -self.ocsvm.decision_function(X)[0]
            
            is_anomaly = prediction == -1
            
            return {
                "is_anomaly": is_anomaly,
                "score": float(abs(score)),
                "method": self.method,
                "triggered_feature": "OCSVM" if is_anomaly else None
            }
        except Exception as e:
            print(f"OCSVM prediction failed: {e}")
            return {
                "is_anomaly": False,
                "score": 0.0,
                "method": self.method,
                "triggered_feature": None
            }
    
    def _predict_isolation_forest(self, speed, distance):
        """Isolation Forest anomaly detection."""
        if self.isolation_forest is None:
            return {
                "is_anomaly": False,
                "score": 0.0,
                "method": self.method,
                "triggered_feature": None
            }
        
        try:
            X = np.array([[speed, distance]])
            prediction = self.isolation_forest.predict(X)[0]
            score = -self.isolation_forest.score_samples(X)[0]
            
            is_anomaly = prediction == -1
            
            return {
                "is_anomaly": is_anomaly,
                "score": float(abs(score)),
                "method": self.method,
                "triggered_feature": "IsolationForest" if is_anomaly else None
            }
        except Exception as e:
            print(f"IsolationForest prediction failed: {e}")
            return {
                "is_anomaly": False,
                "score": 0.0,
                "method": self.method,
                "triggered_feature": None
            }
    
    def reset(self):
        """Reset detector for next video."""
        self.speed_history.clear()
        self.distance_history.clear()
        self.feature_history = []
        self.frame_count = 0
        self.is_trained = False
        self.ocsvm = None
        self.isolation_forest = None
