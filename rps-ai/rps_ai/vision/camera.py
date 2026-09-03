import cv2


class Camera:
    """
    Wrapper around OpenCV VideoCapture.

    Handles:
    - Camera initialization
    - Frame acquisition
    - Horizontal flipping (mirror view)
    - Cleanup
    """

    def __init__(self, device_index: int = 0):
        self.cap = cv2.VideoCapture(device_index)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        # Optional settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def read(self):
        """
        Returns:
            success (bool), frame (numpy.ndarray)
        """
        success, frame = self.cap.read()

        if not success:
            return False, None

        frame = cv2.flip(frame, 1)
        return True, frame

    def release(self):
        """Release webcam resources."""
        self.cap.release()
        cv2.destroyAllWindows()