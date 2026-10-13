import numpy as np
import torch
from collections import Counter, deque

from letter_convert import LetterMLP

LETTERS = [chr(ord('A') + i) for i in range(26)]

class LetterClassification:
    def __init__(self, model_path, window=10, threshold=0.7):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.classes = checkpoint["classes"]
        self.model = LetterMLP(len(self.classes))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.threshold = threshold
        self.history = deque(maxlen=window)

    # def preprocess(self, landmarks):
    #     # first take the landmarks and put it into a numpy vector
    #     pts = np.array(landmarks, dtype=np.float32)
    #     # get rid of the wrist point
    #     pts = pts[1:]
    #     # then we normalize to get the scale of the hand
    #     scale = np.linalg.norm(pts[9])
    #     # given the scale, we divide all the points by it
    #     if pts > 0:
    #         pts /= scale
    #     # then we flatten the points into a single vector
    #     return torch.from_numpy(pts.flatten()).unsqueeze(0)

    def preprocess(self, vector):
        return torch.from_numpy(vector).unsqueeze(0)

    @torch.no_grad()
    def predict(self, vector):
        # first we get the vector representing all the probabilties of each letter
        probablities = torch.softmax(self.model(self.preprocess(vector)), dim=1)[0]
        # then we get the letter with the highest probability and its confidence
        confidence, index = probablities.max(dim=0)
        # based on the confidence, we either return the letter or None if the confidence is too low
        if confidence >= self.threshold:
            letter = LETTERS[index]
        else:
            letter = None
        # then the letter gets appended to the history of letters that showed on the frame
        self.history.append(letter)
        # then we find the letter than appeared most often in the history
        # so we get both the best and the count of the best letter
        best, count = Counter(self.history).most_common(1)[0]
        if count > len(self.history) // 2:
            return best
        else:
            return None


