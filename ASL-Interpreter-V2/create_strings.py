import numpy as np

CLEAR_WORD = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]

class WordBuilder:
    def __init__(self, hold_frames=5, gap_frames=5):
        self.hold_frames = hold_frames
        self.gap_frames = gap_frames
        self.letters = []
        self.candidate = None
        self.candidate_count = 0
        self.empty_count = 0
        self.last_appended = None

    def update(self, letter):
        if letter is None:
            self.empty_count += 1
            self.candidate = None
            self.candidate_count = 0
            if self.empty_count >= self.gap_frames:
                self.last_appended = None
            return

        self.empty_count = 0
        if letter == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = letter
            self.candidate_count = 1

        if self.candidate_count == self.hold_frames and letter != self.last_appended:
            self.letters.append(letter)
            self.last_appended = letter

    def finish(self):
        word = "".join(self.letters)
        self.letters = []
        self.last_appended = None
        return word


def _cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def is_open_five(pts):
    p = np.array([pt[:2] for pt in pts])
    straight = all(_cos(p[pip] - p[mcp], p[tip] - p[pip]) > np.cos(np.radians(25))
                   for mcp, pip, tip in CLEAR_WORD)
    dirs = [p[tip] - p[mcp] for mcp, _, tip in CLEAR_WORD]
    spread = all(_cos(dirs[i], dirs[i + 1]) < np.cos(np.radians(8)) for i in range(3))
    thumb_out = _cos(p[4] - p[2], dirs[0]) < np.cos(np.radians(30))
    return straight and spread and thumb_out