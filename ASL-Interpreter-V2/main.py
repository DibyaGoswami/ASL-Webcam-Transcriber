import csv
import os
import time
import cv2
from landmarks import HandTracker, normalize

DATA_PATH = "data/letters.csv"


def save_sample(vec, label):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "a", newline="") as f:
        csv.writer(f).writerow([label, *vec])


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker()
    start = time.monotonic()
    status = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # VIDEO mode requires strictly increasing timestamps
        hands = tracker.detect(rgb_frame, int((time.monotonic() - start) * 1000))

        h, w = frame.shape[:2]
        for pts, _ in hands:
            for x, y, _ in pts:
                cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)

        cv2.putText(frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        cv2.imshow('Webcam Feed', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if ord('a') <= key <= ord('z'):
            if len(hands) == 1:
                pts, side = hands[0]
                save_sample(normalize(pts, side), chr(key).upper())
                status = f"saved {chr(key).upper()}"
            else:
                status = "need exactly 1 hand"

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()