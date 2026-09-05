import cv2
import numpy as np


def overlay_transparent(background: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
    """Blends a 4-channel BGRA transparent image onto a 3-channel BGR background at position (x, y)."""
    bg_h, bg_w, _ = background.shape
    ol_h, ol_w, channels = overlay.shape

    # If x or y exceed boundaries, clamp bounds
    if x >= bg_w or y >= bg_h or x + ol_w <= 0 or y + ol_h <= 0:
        return background

    # Calculate clipping regions for boundaries
    x1, x2 = max(0, x), min(bg_w, x + ol_w)
    y1, y2 = max(0, y), min(bg_h, y + ol_h)

    overlay_x1, overlay_x2 = max(0, -x), ol_w - max(0, (x + ol_w) - bg_w)
    overlay_y1, overlay_y2 = max(0, -y), ol_h - max(0, (y + ol_h) - bg_h)

    # Extract regions
    bg_crop = background[y1:y2, x1:x2]
    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    if channels == 4:
        # Separate RGB and Alpha channels
        overlay_rgb = overlay_crop[:, :, :3]
        alpha = overlay_crop[:, :, 3] / 255.0

        # Expand dimensions for broad broadcasting
        alpha = np.expand_dims(alpha, axis=2)

        # Blend images using alpha channel as mask
        blended = (1.0 - alpha) * bg_crop + alpha * overlay_rgb
        background[y1:y2, x1:x2] = blended.astype(np.uint8)
    else:
        background[y1:y2, x1:x2] = overlay_crop

    return background