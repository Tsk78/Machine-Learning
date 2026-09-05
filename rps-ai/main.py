import cv2

from rps_ai.vision.camera import Camera
from rps_ai.vision.gesture import GestureDetector


def main():

    camera = Camera()
    detector = GestureDetector()

    while True:

        success, frame = camera.read()

        if not success or frame is None:
            continue

        frame, landmarks = detector.detect(frame)

        if landmarks is not None:
            cv2.putText(
                frame,
                f"Landmarks: {len(landmarks)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Rock Paper Scissors AI", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()
    camera.release()


if __name__ == "__main__":
    main()