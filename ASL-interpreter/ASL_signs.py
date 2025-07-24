import numpy as np
import mediapipe as mp

def is_fist(landmarks):
    """
    YES: Fist (all finger tips close to their MCPs)
    """
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17]
    return all(np.linalg.norm(np.array(landmarks[t][:2]) - np.array(landmarks[m][:2])) < 0.06 for t, m in zip(tips, mcps))

def is_no(landmarks):
    """
    NO: Index & middle extended and touching, others curled
    """
    idx_tip = np.array(landmarks[8][:2])
    mid_tip = np.array(landmarks[12][:2])
    ring_tip = np.array(landmarks[16][:2])
    pinky_tip = np.array(landmarks[20][:2])
    palm = np.array(landmarks[0][:2])
    # Index & middle close, ring & pinky curled
    idx_mid_dist = np.linalg.norm(idx_tip - mid_tip)
    ring_curl = np.linalg.norm(ring_tip - palm)
    pinky_curl = np.linalg.norm(pinky_tip - palm)
    return (idx_mid_dist < 0.04) and (ring_curl < 0.09) and (pinky_curl < 0.09)

def is_thank_you(landmarks, mouth_y):
    """
    THANK YOU: All fingers extended, fingertips near mouth (y small), palm facing out
    """
    extended = all(landmarks[t][1] < landmarks[m][1] for t, m in zip([8,12,16,20],[6,10,14,18]))
    fingertips_y = np.mean([landmarks[i][1] for i in [8,12,16,20]])
    # Mouth_y: top 0.4 of frame
    return extended and (fingertips_y < mouth_y+0.05)

def is_please(landmarks):
    """
    PLEASE: Flat hand, palm out, moves in circular motion on chest 
    (simplified: all fingers extended & close together)
    """
    extended = all(landmarks[t][1] < landmarks[m][1] for t, m in zip([8,12,16,20],[6,10,14,18]))
    tips = [np.array(landmarks[i][:2]) for i in [8,12,16,20]]
    distances = [np.linalg.norm(tips[i] - tips[i+1]) for i in range(3)]
    return extended and all(d < 0.08 for d in distances)

def is_hello(landmarks):
    """
    HELLO: Palm out, all fingers extended, hand near head 
    (simplified: all fingers extended, palm above wrist)
    """
    extended = all(landmarks[t][1] < landmarks[m][1] for t, m in zip([8,12,16,20],[6,10,14,18]))
    wrist_y = landmarks[0][1]
    mean_tip_y = np.mean([landmarks[i][1] for i in [8,12,16,20]])
    return extended and (mean_tip_y < wrist_y - 0.04)

def is_goodbye(landmarks, prev_landmarks):
    """
    GOODBYE: Waving motion (open palm, fingers together, x position oscillates)
    """
    if prev_landmarks is None:
        return False
    extended = all(landmarks[t][1] < landmarks[m][1] for t, m in zip([8,12,16,20],[6,10,14,18]))
    # Oscillation: tip x coords change significantly
    tip_x_now = np.mean([landmarks[i][0] for i in [8,12,16,20]])
    tip_x_prev = np.mean([prev_landmarks[i][0] for i in [8,12,16,20]])
    return extended and abs(tip_x_now - tip_x_prev) > 0.07

def detect_sign(landmarks, prev_landmarks=None, mouth_y=0.20):
    """
    Main function to detect ASL signs from hand landmarks
    
    Args:
        landmarks: List of hand landmark coordinates [(x, y, z), ...]
        prev_landmarks: Previous frame landmarks for motion detection
        mouth_y: Y coordinate of mouth position (default 0.20)
    
    Returns:
        String: Detected sign name or None if no sign detected
    """
    if is_fist(landmarks):
        return "YES"
    elif is_no(landmarks):
        return "NO"
    elif is_thank_you(landmarks, mouth_y):
        return "THANK YOU"
    elif is_please(landmarks):
        return "PLEASE"
    elif is_hello(landmarks):
        return "HELLO"
    elif is_goodbye(landmarks, prev_landmarks):
        return "GOODBYE"
    
    return None