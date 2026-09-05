import os
import csv
import cv2
import numpy as np
from rps_ai.vision.gesture import GestureDetector

DATA_FILE = "rps_dataset.csv"


def normalize_landmarks(landmarks):
    """Normalize 21 landmark coordinates relative to wrist (point 0) and hand scale."""
    if not landmarks or len(landmarks) < 21:
        return None

    pts = np.array([(lm[0], lm[1]) for lm in landmarks], dtype=np.float32)

    # 1. Translate wrist (index 0) to origin (0, 0)
    pts -= pts[0]

    # 2. Scale by hand size (distance from wrist to middle MCP joint, index 9)
    scale = np.linalg.norm(pts[9])
    if scale > 0:
        pts /= scale

    # Flatten (21, 2) -> (42,) feature vector
    return pts.flatten().tolist()


def main():
    cap = cv2.VideoCapture(0)
    detector = GestureDetector()

    # Ensure CSV file header exists
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            header = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["label"]
            writer.writerow(header)

    print("Data Collection Running!")
    print("Press 'r' for ROCK | 'p' for PAPER | 's' for SCISSORS | 'q' to QUIT")

    counts = {"ROCK": 0, "PAPER": 0, "SCISSORS": 0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame, landmarks = detector.detect(frame)
        norm_features = normalize_landmarks(landmarks)

        key = cv2.waitKey(1) & 0xFF
        label = None

        if key == ord('r'):
            label = "ROCK"
        elif key == ord('p'):
            label = "PAPER"
        elif key == ord('s'):
            label = "SCISSORS"
        elif key == ord('q'):
            break

        if label and norm_features:
            # Separate X and Y coordinates into feature list
            xs = norm_features[0::2]
            ys = norm_features[1::2]
            row = xs + ys + [label]

            with open(DATA_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            counts[label] += 1
            print(f"Recorded {label}! Total counts: {counts}")

        # Display instructions
        cv2.putText(
            frame,
            f"R: {counts['ROCK']} | P: {counts['PAPER']} | S: {counts['SCISSORS']}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.imshow("Data Collector", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()