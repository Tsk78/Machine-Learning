import cv2

from rps_ai.vision.camera import Camera


def main():
    camera = Camera()

    while True:
        success, frame = camera.read()

        if not success:
            break

        cv2.imshow("Rock Paper Scissors AI", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()


if __name__ == "__main__":
    main()