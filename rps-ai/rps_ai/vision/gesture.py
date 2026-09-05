import os
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

# 21 Hand keypoint skeleton connections
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),        # Thumb
    (0,5), (5,6), (6,7), (7,8),        # Index
    (0,9), (9,10), (10,11), (11,12),   # Middle
    (0,13), (13,14), (14,15), (15,16), # Ring
    (0,17), (17,18), (18,19), (19,20)  # Pinky
]

BEST_WEIGHTS_PATH = "runs/pose/runs/hand_pose/yolo11n_hand-3/weights/best.pt"


class GestureDetector:
    """Native hand landmark detector powered by fine-tuned YOLO11n weights."""

    def __init__(self, model_path: str = BEST_WEIGHTS_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weights file not found at '{model_path}'. Check path.")
        self.model = YOLO(model_path)

    def detect(self, frame):
        raw_results = self.model([frame], conf=0.35, verbose=False)
        results_list = list(raw_results) if not isinstance(raw_results, list) else raw_results

        if not results_list:
            return frame, None

        first_result = results_list[0]

        if not isinstance(first_result, Results):
            return frame, None

        if first_result.keypoints is None or len(first_result.keypoints) == 0:
            return frame, None

        # Extract landmark coordinates
        keypoint_data = first_result.keypoints.data[0].cpu().numpy()
        landmarks = []

        for kp in keypoint_data:
            x, y = int(kp[0]), int(kp[1])
            landmarks.append((x, y, 0.0))
            if x > 0 and y > 0:
                # Keypoint visual marker
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # Draw hand skeleton connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                pt1 = (landmarks[start_idx][0], landmarks[start_idx][1])
                pt2 = (landmarks[end_idx][0], landmarks[end_idx][1])
                if pt1[0] > 0 and pt2[0] > 0:
                    cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

        return frame, landmarks if len(landmarks) == 21 else None

    def close(self):
        pass