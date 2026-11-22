"""Give every piece a fingerprint, so its identity never depends on where it was left.

The failure this fixes: a piece was tracked as "the slot it came from".  When a release did
not land it back in that slot -- an occupied slot, a stale box, a piece landing on the board --
the bookkeeping stayed wrong and the piece could not be found again.  Its cell was unfillable
for the rest of the game.

A fingerprint is the piece's own pixels, reduced to a 32x24 normalised grey template:

    same piece, same view        matchTemplate TM_CCOEFF_NORMED  ~0.99
    different piece, same class  typically 0.2-0.6

so "where is this piece?" becomes an image search, and "did it land where I aimed?" becomes a
comparison against the piece itself instead of "is something bright here".

  tpl = fingerprint.template(img, box)
  fingerprint.find(img, tpl)          -> (x, y, score) over the whole window
  fingerprint.score_at(img, tpl, box) -> how well that box still shows the same piece
"""
from __future__ import annotations

import cv2
import numpy as np

TW, TH = 32, 24          # template size
GOOD = 0.80              # score that means "this is that piece"


def _crop(img: np.ndarray, box) -> np.ndarray:
    x, y, w, h = box
    return img[max(0, y):y + h, max(0, x):x + w]


def template(img: np.ndarray, box) -> np.ndarray | None:
    """Normalised grey thumbnail of the piece inside `box`."""
    crop = _crop(img, box)
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (TW, TH), interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32)
    g -= g.mean()
    s = float(g.std())
    if s < 1.0:
        return None               # flat patch: nothing to recognise
    return g / s


def score_at(img: np.ndarray, tpl: np.ndarray, box) -> float:
    """How much the piece inside `box` still looks like `tpl`."""
    if tpl is None:
        return 0.0
    crop = _crop(img, box)
    if crop.size == 0:
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (TW, TH), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= g.mean()
    s = float(g.std())
    if s < 1.0:
        return 0.0
    g /= s
    return float((g * tpl).mean())


def find(img: np.ndarray, tpl: np.ndarray, regions: list[tuple[int, int, int, int]] | None = None,
         box_size: tuple[int, int] | None = None) -> tuple[int, int, float]:
    """Best match for `tpl` anywhere in `regions` (default: the whole window).

    The screenshot is shrunk by the same factor the template was shrunk by, so a 112x85 piece
    becomes the 32x24 template at the right scale.  Upscaling the template instead throws away
    its high frequencies and the peak collapses (measured: < 0.5 at the true position).
    """
    if tpl is None:
        return (-1, -1, 0.0)
    w, h = box_size or (112, 85)
    sx, sy = w / TW, h / TH
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (max(1, int(grey.shape[1] / sx)), max(1, int(grey.shape[0] / sy))),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    best = (-1, -1, -1.0)
    for (x0, y0, rx, ry) in (regions or [(0, 0, grey.shape[1], grey.shape[0])]):
        a0, b0 = int(x0 / sx), int(y0 / sy)
        a1, b1 = int((x0 + rx) / sx), int((y0 + ry) / sy)
        a0, b0 = max(0, a0), max(0, b0)
        a1, b1 = min(small.shape[1], a1), min(small.shape[0], b1)
        if a1 - a0 <= TW or b1 - b0 <= TH:
            continue
        res = cv2.matchTemplate(small[b0:b1, a0:a1], tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _mn, mx, _ml, ml = cv2.minMaxLoc(res)
        if mx > best[2]:
            best = (int((a0 + ml[0]) * sx), int((b0 + ml[1]) * sy), float(mx))
    return best


def where(img: np.ndarray, tpl: np.ndarray, tray_boxes: list[tuple] | None = None,
          preferred: tuple | None = None) -> tuple[str, tuple | None, float]:
    """First ask the slots, then the whole window.

    Returns ('slot', box, score) when the piece is sitting in a known slot, ('loose', (x,y,w,h),
    score) when it is somewhere else, or ('lost', None, score).
    """
    if tpl is None:
        return ("lost", None, 0.0)
    if preferred is not None and score_at(img, tpl, preferred) >= GOOD:
        return ("slot", preferred, score_at(img, tpl, preferred))
    if tray_boxes:
        scores = [(score_at(img, tpl, b), b) for b in tray_boxes]
        scores.sort(reverse=True)
        if scores and scores[0][0] >= GOOD:
            return ("slot", scores[0][1], scores[0][0])
    x, y, s = find(img, tpl, box_size=(112, 85))
    if s >= GOOD:
        return ("loose", (x, y, 112, 85), s)
    return ("lost", None, s)
