import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

WRIST, MIDDLE_MCP = 0, 9

class HandTracker:
    def __init__(self, model_path="hand_landmarker.task", num_hands=2, running_mode=vision.RunningMode.VIDEO):
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_hands=num_hands,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb_frame, timestamp_ms=None):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        if timestamp_ms is None:
             result = self.landmarker.detect(image)
        else: 
            result = self.landmarker.detect_for_video(image, timestamp_ms)
        hands = []
        for lms, handedness in zip(result.hand_landmarks, result.handedness):
            pts = np.array([(lm.x, lm.y, lm.z) for lm in lms], dtype=np.float32)
            hands.append((pts, handedness[0].category_name))
        return hands

    def close(self):
        self.landmarker.close()


def normalize(pts, side):
    # Position, distance from camera, and which hand is used shouldn't affect the label
    out = pts - pts[WRIST]
    if side == "Left":
        out[:, 0] *= -1
    v = out[MIDDLE_MCP, :2].copy()
    scale = np.linalg.norm(v)
    if scale > 0:
        out /= scale
        v /= scale
    # Rotate so wrist->middle MCP points up; shape features become tilt-invariant
    R = np.array([[-v[1], v[0]], [-v[0], -v[1]]])
    out[:, :2] = out[:, :2] @ R.T
    # Keep original direction: H/U, K/P, G/Q differ only by orientation
    return np.concatenate([out.flatten(), v])
