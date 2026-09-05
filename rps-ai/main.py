import cv2
from rps_ai.vision.gesture import GestureDetector
from rps_ai.vision.classifier import GestureClassifier


def main():
    cap = cv2.VideoCapture(0)
    detector = GestureDetector()
    classifier = GestureClassifier()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for intuitive mirrored camera display
        frame = cv2.flip(frame, 1)

        # Detect keypoints and draw skeleton
        frame, landmarks = detector.detect(frame)

        # Predict gesture
        gesture = classifier.classify(landmarks)

        # Render prediction box
        color = (0, 255, 0) if gesture in ["ROCK", "PAPER", "SCISSORS"] else (0, 0, 255)
        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
            cv2.LINE_AA
        )

        cv2.imshow("RPS AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()