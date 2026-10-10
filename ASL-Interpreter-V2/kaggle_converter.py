import csv
import os
import cv2
from mediapipe.tasks.python import vision
from landmarks import HandTracker, normalize

SRC_PATH = "data/raw/asl_alphabet_train/asl_alphabet_train"
OUTPUT_PATH = "data/converted/letters.csv"

def main():
    tracker = HandTracker(num_hands=1, running_mode=vision.RunningMode.IMAGE)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        for label in sorted(os.listdir(SRC_PATH)):
            # Skip del/nothing/space
            if len(label) != 1:
                continue
            class_dir = os.path.join(SRC_PATH, label)
            saved = missed = 0
            for name in os.listdir(class_dir):
                bgr = cv2.imread(os.path.join(class_dir, name))
                if bgr is None:
                    continue
                hands = tracker.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if len(hands) != 1:
                    missed += 1
                    continue
                pts, side = hands[0]
                writer.writerow([label, *normalize(pts, side)])
                saved += 1
            print(f"{label}: saved {saved}, no hand found in {missed}")

    tracker.close()


if __name__ == "__main__":
    main()