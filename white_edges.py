"""Read a pure-white piece's four cuts from its own silhouette.

Why this module exists: the photo edge reader cannot see a white-on-white edge -- on a live board
one white piece came back with all four sides unknown -- but a white piece is a bright blob on a
dark tray, so its outline needs no tuning at all.

Why it matters: the five white pieces are indistinguishable by picture.  The panel cells that
belong to them are plain white tiles (mean 255, no drawing), and an empty board cell renders no
shape either (measured: flat board colour 23.0 at every blank cell, identical to the gaps between
cells).  So the only thing a white piece carries is its own cut, and until now the loop had nothing
to sort them by -- each wrong guess costs two moves, and on one recorded run 22 of ~30 refusals were
white-into-white.

What it does not do: decide.  Agreement with the photo reader on sides both could read was 9 of 14,
so these readings order the candidates and the engine's snap still has the last word.

The reader is self-calibrating: no core rectangle is assumed.  For each side the mask's own profile
gives the core line (its median), and the cut is what the middle third does against it -- a tab
pulls the profile outward, a notch pushes it inward.
"""
from __future__ import annotations

import cv2
import numpy as np

import vision

SIDES = ("top", "right", "bottom", "left")

TAB_PX = 9        # the engine's tab is ~19 px; half of it is a safe "this is a cut" threshold
FLAT_PX = 5


def white_mask(img: np.ndarray, box, pad: int = 26) -> np.ndarray:
    x, y, w, h = box
    crop = vision.sub(img, (x - pad, y - pad, w + 2 * pad, h + 2 * pad))
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
    m = ((g > 198) & (sat < 40)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def read_white_edges(img: np.ndarray, box) -> dict:
    """{'top': 'tab'|'notch'|'flat'|None, ...} from the piece's silhouette."""
    m = white_mask(img, box)
    if int(m.sum()) < 500:
        return {s: None for s in SIDES}
    ys, xs = np.nonzero(m)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    out: dict[str, str | None] = {}

    def profile(side: str) -> np.ndarray:
        if side == "top":
            return np.argmax(m, axis=0).astype(np.float32)
        if side == "bottom":
            return (h - 1 - np.argmax(m[::-1], axis=0)).astype(np.float32)
        if side == "left":
            return np.argmax(m, axis=1).astype(np.float32)
        return (w - 1 - np.argmax(m[:, ::-1], axis=1)).astype(np.float32)

    for side in SIDES:
        prof = profile(side)
        n = len(prof)
        mid = prof[n // 3: 2 * n // 3]
        core = float(np.median(prof))
        if side in ("top", "left"):
            outward, inward = core - float(mid.min()), float(mid.max()) - core
        else:
            outward, inward = float(mid.max()) - core, core - float(mid.min())
        if outward > TAB_PX:
            out[side] = "tab"
        elif inward > TAB_PX:
            out[side] = "notch"
        elif abs(outward) < FLAT_PX and abs(inward) < FLAT_PX:
            out[side] = "flat"
        else:
            out[side] = None
    return out


def flats(out: dict) -> set[str]:
    return {s for s, v in out.items() if v == "flat"}
