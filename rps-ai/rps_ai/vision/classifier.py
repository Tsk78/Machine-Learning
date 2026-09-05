import os
import pickle
from collections import Counter, deque
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


class GestureClassifier:
    """Classifies 21-point hand landmarks using a trained ML model."""

    def __init__(self, model_path: str = "rps_classifier.pkl", window_size: int = 7):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. Run 'python train_classifier.py' first."
            )
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Match column names used during dataset creation
        self.feature_names = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
        
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def _normalize(self, landmarks: List[Tuple[int, int, float]]) -> Optional[pd.DataFrame]:
        pts = np.array([(lm[0], lm[1]) for lm in landmarks], dtype=np.float32)

        # Translate wrist to origin
        pts -= pts[0]

        # Scale by distance from wrist to middle MCP
        scale = np.linalg.norm(pts[9])
        if scale == 0:
            return None
        pts /= scale

        # Flatten into [x0...x20, y0...y20]
        xs = pts[:, 0]
        ys = pts[:, 1]
        features_flat = np.hstack([xs, ys]).reshape(1, -1)

        # Convert to DataFrame with feature names to suppress sklearn UserWarning
        return pd.DataFrame(features_flat, columns=self.feature_names)

    def classify(self, landmarks: Optional[List[Tuple[int, int, float]]]) -> str:
        if not landmarks or len(landmarks) < 21:
            self.history.append("NO_HAND")
            return self._get_majority_prediction()

        features_df = self._normalize(landmarks)
        if features_df is None:
            self.history.append("UNKNOWN")
            return self._get_majority_prediction()

        # Predict using named DataFrame
        raw_pred = self.model.predict(features_df)[0]
        self.history.append(str(raw_pred))

        return self._get_majority_prediction()

    def _get_majority_prediction(self) -> str:
        if not self.history:
            return "NO_HAND"
        counts = Counter(self.history)
        most_common, _ = counts.most_common(1)[0]
        return most_common

    def reset_buffer(self):
        self.history.clear()