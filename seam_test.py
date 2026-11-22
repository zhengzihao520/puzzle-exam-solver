"""Can two pieces be paired by image continuity alone?  Offline ground truth first.

The engine cuts a photo into cells; the pixels either side of a shared edge are *adjacent in
the source*, so a correct pair continues smoothly across the seam and a wrong pair does not.
That would give the assembly for free -- no piece has to be lifted, no cut has to be measured --
which is the only way to get near the exam's 25-move budget.

This file answers one question before anything is built on top of it: given pieces cut and
scrambled exactly the way the site does it, does the seam score put the true neighbour first?

Sampling rules that matter (both learned the hard way):
  * stay 4 px inside each piece: the engine strokes the outline with 40% white, 2 px wide,
  * skip the middle third of the edge: that is where the tab or notch is, and its geometry
    would dominate the comparison,
  * compare like with like across the seam, because a correct pair is a few source pixels apart.

  python seam_test.py [--trials 12]
"""
from __future__ import annotations

import argparse
import glob
import random
from collections import Counter

import cv2
import numpy as np

import piececutter as pc

W, H = 112, 85
N = 5
INSET = 4


def cut_photo(photo: np.ndarray, edges: dict[tuple[int, int], list[int]]) -> dict:
    """Pieces exactly as the engine builds them: the source window around each cell, masked."""
    tab, margin, cw, ch = pc.geometry(W, H)
    hh, ww = photo.shape[:2]
    pieces = {}
    for (r, c), e in edges.items():
        mask = pc.piece_mask(tuple(e), W, H)                 # (ch, cw) silhouette
        ox, oy = c * W, r * H                                # cell origin in the photo
        win = np.zeros((ch, cw, 3), np.uint8)
        x0, y0 = ox - int(round(margin)), oy - int(round(margin))
        for yy in range(ch):                                 # copy with clamping
            sy = min(max(y0 + yy, 0), hh - 1)
            for xx in range(cw):
                sx = min(max(x0 + xx, 0), ww - 1)
                win[yy, xx] = photo[sy, sx]
        win[mask == 0] = 0
        pieces[(r, c)] = win
    return pieces


def edge_strips(canvas: np.ndarray, side: str):
    """The usable part of one edge: outer thirds only, inset from the stroke."""
    tab, margin, cw, ch = pc.geometry(W, H)
    x0, y0 = int(round(margin)), int(round(margin))
    if side == "right":
        col = x0 + W - 1 - INSET
        strip = canvas[y0:y0 + H, col]
    elif side == "left":
        col = x0 + INSET
        strip = canvas[y0:y0 + H, col]
    elif side == "bottom":
        row = y0 + H - 1 - INSET
        strip = canvas[row, x0:x0 + W]
    else:
        row = y0 + INSET
        strip = canvas[row, x0:x0 + W]
    n = len(strip)
    third = n // 3
    return np.concatenate([strip[:third], strip[n - third:]])     # drop the tab/notch zone


def seam(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return 1e9
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    args = ap.parse_args()

    photos = sorted(glob.glob(str(pc.__file__).replace("piececutter.py", "")
                              + r"raw_assets\*\*.jpg"))
    if not photos:
        photos = sorted(glob.glob(r"F:\exam_next\raw_assets\*\*.jpg"))
    print(f"source photos: {len(photos)}")
    rng = random.Random(7)
    photo = cv2.imread(rng.choice(photos))
    photo = cv2.resize(photo, (W * N, H * N), interpolation=cv2.INTER_AREA)

    hits_r, hits_b, trials = 0, 0, 0
    margins = []
    for _ in range(1):
        # a legal 5x5 edge matrix, exactly as the engine generates it
        h = [[rng.choice((-1, 1)) for _ in range(N)] for _ in range(N - 1)]
        v = [[rng.choice((-1, 1)) for _ in range(N - 1)] for _ in range(N)]
        edges = {}
        for r in range(N):
            for c in range(N):
                edges[(r, c)] = [
                    0 if r == 0 else -h[r - 1][c],            # top
                    0 if c == N - 1 else v[r][c],             # right
                    0 if r == N - 1 else h[r][c],             # bottom
                    0 if c == 0 else -v[r][c - 1],            # left
                ]
        pieces = cut_photo(photo, edges)
        keys = list(pieces)
        for r in range(N):
            for c in range(N):
                trials += 1
                a = pieces[(r, c)]
                if c < N - 1:                                  # true right neighbour
                    mine = edge_strips(a, "right")
                    scores = {k: seam(mine, edge_strips(pieces[k], "left"))
                              for k in keys if k != (r, c)}
                    best = min(scores, key=scores.get)
                    hits_r += (best == (r, c + 1))
                    if best == (r, c + 1):
                        others = sorted(v for k, v in scores.items() if k != (r, c + 1))
                        margins.append(others[0] - scores[(r, c + 1)])
                if r < N - 1:                                  # true bottom neighbour
                    mine = edge_strips(a, "bottom")
                    scores = {k: seam(mine, edge_strips(pieces[k], "top"))
                              for k in keys if k != (r, c)}
                    best = min(scores, key=scores.get)
                    hits_b += (best == (r + 1, c))

    pairs_r = sum(1 for r in range(N) for c in range(N) if c < N - 1)
    pairs_b = sum(1 for r in range(N) for c in range(N) if r < N - 1)
    print(f"right-neighbour chosen first: {hits_r}/{pairs_r}  ({hits_r / pairs_r:.0%})")
    print(f"below-neighbour chosen first: {hits_b}/{pairs_b}  ({hits_b / pairs_b:.0%})")
    if margins:
        m = sorted(margins)
        print(f"margin over the next best candidate: min {m[0]:.1f}  median {m[len(m)//2]:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
