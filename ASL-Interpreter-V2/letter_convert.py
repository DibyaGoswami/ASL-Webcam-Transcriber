from collections import Counter
import csv
import os
import numpy as np
import torch
from torch import nn

CSV_PATH = "data/converted/letters.csv"
MODEL_PATH = "models/letters.pt"
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_ITERS = 6000
EVAL_ITERS = 50
device = torch.device("cuda:1")
# device = "cuda" if torch.cuda.is_available() else "cpu"

class LetterMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(65, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def data_loader(paths):
    letter, vector = [], []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.reader(f):
                letter.append(row[0])
                vector.append([float(x) for x in row[1:]])
    return np.array(letter), np.array(vector, dtype=np.float32)

def split(y, test_frac=0.2, seed=4):
    idx = np.random.default_rng(seed).permutation(len(y))  
    n_test = int(len(y) * test_frac)
    return idx[n_test:], idx[:n_test]

def get_batch(data, split):
    vec, lab = data[split]
    ix = torch.randint(len(lab), (BATCH_SIZE,))
    x, y = vec[ix].to(device), lab[ix].to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, loss_fn, data):
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(data, split)
            logits = model(X)
            losses[k] = loss_fn(logits, Y).item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    labels, X = data_loader([CSV_PATH])
    classes = sorted(set(labels))
    y = []
    for l in labels:
        y.append(classes.index(l))
    y = np.array(y)

    train_idx, test_idx = split(y)

    # so our X and our y are arrays, we need to convert them into tensors
    X_ten = torch.from_numpy(X)
    y_ten = torch.from_numpy(y)

    data = {
        "train": (X_ten[train_idx], y_ten[train_idx]),
        "test": (X_ten[test_idx], y_ten[test_idx]),
    }

    # now we basically initialize our model and set up the training loop
    model = LetterMLP(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(MAX_ITERS):
        if step % EVAL_ITERS == 0:
            stats = estimate_loss(model, loss_fn, data)
            print(f'step: {step}, train loss: {stats["train"]:.4f}, validation loss: {stats["test"]:.4f}')
        xb, yb = get_batch(data, "train")
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    stat = estimate_loss(model, loss_fn, data)
    print(f'final train loss: {stat["train"]:.4f}, final validation loss: {stat["test"]:.4f}')

    model.eval()
    with torch.no_grad():
        X_test, y_test = data["test"]
        X_test = X_test.to(device)
        logits = model(X_test)
        predictions = logits.argmax(dim=1).cpu()

    mistakes = Counter()
    true_list = y_test.tolist()
    predictions_list = predictions.tolist()
    for i in range(len(true_list)):
        t = true_list[i]
        p = predictions_list[i]
        if t != p:
            mistakes[(classes[t], classes[p])] += 1

    for (t, p), count in mistakes.most_common(10):
        print(f"{t} predicted as {p}: {count}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": [str(c) for c in classes]}, MODEL_PATH)


if __name__ == "__main__":
       main()