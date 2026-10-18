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
    # DirectShow avoids MSMF stream-start failures and slow startup on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera (is another app using it?)")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker()
    builder = WordBuilder()
    start = time.monotonic()
    last_ts = -1
    status = ""
    prediction = ""

    frame_times = []
    mp_times = []
    mlp_times = []
    perf_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        t0 = time.perf_counter()
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # VIDEO mode requires strictly increasing timestamps
        ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
        last_ts = ts
        t_mp = time.perf_counter()
        hands = tracker.detect(rgb_frame, ts)
        mp_times.append(time.perf_counter() - t_mp)

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
                vec = normalize(pts, side) # all this is doing is seperating the steps between predicting and normalizing to measure prediction timing without including time taken to normalize
                t_mlp = time.perf_counter()
                prediction = classifier.predict(vec) or ""
                mlp_times.append(time.perf_counter() - t_mlp)
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
        frame_times.append(time.perf_counter() - t0)
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

    cap.release()
    cv2.destroyAllWindows()
    total = time.perf_counter() - perf_start
    if frame_times:
        print(f"Avg latency: {1000 * sum(frame_times) / len(frame_times):.1f} ms")
        print(f"End-to-end FPS: {len(frame_times) / total:.1f}")
        print(f"Hand tracking: {1000 * sum(mp_times) / len(mp_times):.1f} ms")
    if mlp_times:
        print(f"Letter model: {1000 * sum(mlp_times) / len(mlp_times):.2f} ms")

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()