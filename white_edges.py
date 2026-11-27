"""Read a pure-white piece's four cuts from its own silhouette.

A white piece is a bright blob on a dark tray, so its outline needs no tuning: for each side the
mask's own profile gives the core line (its median), and the cut is what the middle third does
against it -- a tab pulls the profile outward, a notch pushes it inward.  No core rectangle is
assumed, which matters because the tray slot box is not the piece's core.

Used for one thing only: the *flat* sides.  `vision.shape_fit` in the solver counts how a piece's
flat sides line up with the board's edge (a corner wants two, an edge piece one, an interior piece
none), and white pieces had nothing to contribute because a white-on-white edge is invisible to the
photo reader.  Tabs and notches are also returned, and are deliberately not used any more: measured
against the pieces that actually locked into a cell, a deduction built on them agreed on 10 of 12
known sides -- good enough to order candidates, not good enough to decide, and as a filter it cost
more moves than it saved.
"""
from __future__ import annotations

import cv2
import numpy as np

import vision

SIDES = ("top", "right", "bottom", "left")

TAB_PX = 9        # the engine's tab is ~19 px; half of it is a safe "this is a cut" threshold
FLAT_PX = 5
PAD = 26          # wider than a tab (~19 px) so a protruding cut is never clipped
# Brightness floor for the mask.  This started at 198, which was wrong in a way that looked like a
# reader bug: the tabs sit in their own shadow and measure 100-190 grey, so a 198 floor cut them off
# and the mask came out as the 112x85 core rectangle -- every side then read "flat", and a piece
# with tabs all round looked like a plain tile.  Measured on a live tray: the core blob's bounding
# box is 112x85 at 198, 132x136 at 110-80, and the tray background is 23-40, so 110 is far from it.
WHITE_FLOOR = 110
WHITE_SAT = 60


def white_mask(img: np.ndarray, box, pad: int = PAD) -> np.ndarray:
    x, y, w, h = box
    crop = vision.sub(img, (x - pad, y - pad, w + 2 * pad, h + 2 * pad))
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
    m = ((g > WHITE_FLOOR) & (sat < WHITE_SAT)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Keep only the blob the piece's centre is in.  The pad must be wider than a tab, while tray
    # slots are only ~18 px apart, so a neighbour's edge lands inside the crop; without this the
    # profile invents a tab or a notch on a side that is really flat (measured: "wanted flat,
    # locking piece shows notch" on a bottom-row cell, which cannot happen).
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        cy, cx = m.shape[0] // 2, m.shape[1] // 2
        keep = lab[cy, cx] if lab[cy, cx] else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == keep).astype(np.uint8)
    return m


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
