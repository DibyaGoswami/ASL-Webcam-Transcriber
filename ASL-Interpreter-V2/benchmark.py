import csv
import os
import time
import numpy as np
from letter_signs import LetterClassification
from collections import defaultdict

classifier = LetterClassification("models/letters.pt")

with open("data/converted/letters.csv") as f:
    row = next(csv.reader(f))
vec = np.array(row[1:], dtype=np.float32)

for _ in range(100):
    classifier.predict(vec)

n = 1000
t = time.perf_counter()
for _ in range(n):
    classifier.predict(vec)
print(f"{(time.perf_counter() - t) / n * 1e6:.1f} µs per prediction")

correct = defaultdict(int)
total = defaultdict(int)
with open("data/converted/letters.csv") as f:
    for row in csv.reader(f):
        label, vec = row[0], np.array(row[1:], dtype=np.float32)
        total[label] += 1
        correct[label] += classifier.predict(vec) == label

print(f"Test accuracy: {sum(correct.values()) / sum(total.values()):.2%}")
for letter in sorted(total):
    print(letter, f"{correct[letter] / total[letter]:.2%}")

print(sum(p.numel() for p in classifier.model.parameters()), "parameters")
print(f"{os.path.getsize('models/letters.pt') / 1024:.1f} KB")