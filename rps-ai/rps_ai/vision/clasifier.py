import math
from typing import List, Tuple, Optional


class GestureClassifier:
    """Classifies 21-point hand landmarks into Rock, Paper, or Scissors."""

    # Keypoint Indices
    WRIST = 0
    THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
    INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
    MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
    RING_TIP, RING_PIP, RING_MCP = 16, 14, 13
    PINKY_TIP, PINKY_PIP, PINKY_MCP = 20, 18, 17

    def _euclidean_dist(self, p1: Tuple[int, int, float], p2: Tuple[int, int, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _is_finger_extended(
        self,
        landmarks: List[Tuple[int, int, float]],
        tip_idx: int,
        pip_idx: int,
        mcp_idx: int
    ) -> bool:
        """A finger is extended if its tip is farther from the wrist than its PIP/MCP joints."""
        wrist = landmarks[self.WRIST]
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]

        # Compare distances from wrist
        dist_tip_wrist = self._euclidean_dist(tip, wrist)
        dist_pip_wrist = self._euclidean_dist(pip, wrist)

        return dist_tip_wrist > dist_pip_wrist

    def _is_thumb_extended(self, landmarks: List[Tuple[int, int, float]]) -> bool:
        """Special thumb check comparing distance to the pinky base (MCP)."""
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        pinky_mcp = landmarks[self.PINKY_MCP]

        return self._euclidean_dist(thumb_tip, pinky_mcp) > self._euclidean_dist(thumb_ip, pinky_mcp)

    def classify(self, landmarks: Optional[List[Tuple[int, int, float]]]) -> str:
        """Evaluates extended fingers and returns 'ROCK', 'PAPER', 'SCISSORS', or 'UNKNOWN'."""
        if not landmarks or len(landmarks) < 21:
            return "NO_HAND"

        # Check state of each finger
        index_open = self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP)
        middle_open = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)
        ring_open = self._is_finger_extended(landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP)
        pinky_open = self._is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP)

        # Count extended main fingers
        extended_fingers = [index_open, middle_open, ring_open, pinky_open]
        open_count = sum(extended_fingers)

        # Classification Rules
        if open_count == 0:
            return "ROCK"
        elif index_open and middle_open and not ring_open and not pinky_open:
            return "SCISSORS"
        elif open_count >= 3:
            return "PAPER"

        return "UNKNOWN"