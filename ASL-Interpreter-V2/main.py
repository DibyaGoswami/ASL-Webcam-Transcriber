import csv
import os
import time
import cv2
from landmarks import HandTracker, normalize
from letter_signs import LetterClassification
from create_strings import WordBuilder, is_open_five

DATA_PATH = "data/letters.csv"


def save_sample(vec, label):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "a", newline="") as f:
        csv.writer(f).writerow([label, *vec])


def main():
    classifier = LetterClassification("models/letters.pt")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker()
    builder = WordBuilder()
    start = time.monotonic()
    last_ts = -1
    status = ""
    prediction = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # VIDEO mode requires strictly increasing timestamps
        ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
        last_ts = ts
        hands = tracker.detect(rgb_frame, ts)

        h, w = frame.shape[:2]
        for pts, _ in hands:
            for x, y, _ in pts:
                cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)

        # so for when only one hand is up, for that's how the model has been trained so far, we will use letters.pt
        if len(hands) == 1:
            pts, side = hands[0]
            if is_open_five(pts):
                prediction = "CLEAR"
            else:
                prediction = classifier.predict(normalize(pts, side)) or ""
        else:
            prediction = ""

        if prediction == "CLEAR":
            builder.finish()
        else:
            builder.update(prediction or None)

        cv2.putText(frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        cv2.putText(frame, prediction, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        cv2.putText(frame, "".join(builder.letters), (10, 180),
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