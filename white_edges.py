"""Read a pure-white piece's cuts from its own silhouette -- and the shape an empty cell needs.

A white piece is a bright blob on a dark tray, so its outline needs no tuning: for each side the
mask's own profile gives the core line (its median), and the cut is what the middle third does
against it -- a tab pulls the profile outward, a notch pushes it inward.  No core rectangle is
assumed, which matters because the tray slot box is not the piece's core.

Three callers, and they say how the evidence turned out:

* `read_white_edges` -- a piece in the tray.  Tabs and notches are reported, but only the *flat*
  sides are used to order candidates: measured against the pieces that actually locked into a cell,
  a deduction built on the tab/notch labels agreed on 10 of 12 known sides, which is enough to order
  and not enough to decide.
* `read_hole_edges` -- the shape a cell still needs, read from the hole itself (the one dark region
  on a board of bright pieces).  This is the measurement that was missing before: reading the
  *neighbours* and complementing them is a weaker version of the same question.
* `profiles_of` -- the continuous numbers behind both, for pairing a hole with a piece: one wrong
  three-way label would break an exact one-to-one match, a few pixels of noise does not.
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

HOLE_DARK = 60    # the hole is tray grey (23); every placed piece is brighter than this


def white_mask(img: np.ndarray, box, pad: int = PAD) -> np.ndarray:
    x, y, w, h = box
    crop = vision.sub(img, (x - pad, y - pad, w + 2 * pad, h + 2 * pad))
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
    m = ((g > WHITE_FLOOR) & (sat < WHITE_SAT)).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
    # Keep only the blob the piece's centre is in.  The pad must be wider than a tab, while tray
    # slots are only ~18 px apart, so a neighbour's edge lands inside the crop; without this the
    # profile invents a tab or a notch on a side that is really flat (measured: "wanted flat,
    # locking piece shows notch" on a bottom-row cell, which cannot happen).
    #
    # Two white pieces side by side are 18 px apart and a tab is 19 px long, so their blobs can
    # actually touch.  When they do, one component is two pieces wide and every profile is
    # nonsense -- measured on a live tray as "5 white pieces, 1 flat side readable" where 20 sides
    # exist.  Eroding first breaks a bridge that thin; the kept blob is dilated back so its outline
    # is the piece's own again.
    thin = cv2.erode(m, k3, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(thin, 8)
    if n > 1:
        cy, cx = m.shape[0] // 2, m.shape[1] // 2
        keep = lab[cy, cx] if lab[cy, cx] else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = cv2.dilate((lab == keep).astype(np.uint8), k3, iterations=2)
    return m


def _sides_of(m: np.ndarray) -> dict:
    """{'top'|'right'|'bottom'|'left': 'tab'|'notch'|'flat'|None} from a filled silhouette."""
    if int(m.sum()) < 200:
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


def read_white_edges(img: np.ndarray, box) -> dict:
    """{'top': 'tab'|'notch'|'flat'|None, ...} from the piece's silhouette."""
    m = white_mask(img, box)
    return _sides_of(m)


def read_hole_edges(img: np.ndarray, cell_box, pad: int = 24) -> dict:
    """The shape an *empty* cell needs, read from the hole itself.

    The hole is the one region of the board that is not a piece, and its own darkness separates it
    from everything else: take the dark connected component that contains the cell's centre, and
    what is left is exactly the silhouette of the piece that goes there -- including its tabs, which
    reach into the neighbouring cells and are dark there for the same reason.

    Verified against what the board states rather than against a log: a cell in row 0 has a flat
    top, row 4 a flat bottom, column 0 a flat left, column 4 a flat right, and this reader returns
    flat there in 4 of 4 such sides over twelve frames of a recorded Part 3 game
    (`tools/hole_test.py`).
    """
    x, y, w, h = cell_box
    crop = vision.sub(img, (x - pad, y - pad, w + 2 * pad, h + 2 * pad))
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return {s: None for s in SIDES}
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # The white stroke the page draws round the hole is not part of the shape, so it is dropped by
    # the same threshold: it is bright, the hole is dark.
    m = (g < HOLE_DARK).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    cy, cx = pad + h // 2, pad + w // 2       # the cell's centre, in crop coordinates
    if cy >= m.shape[0] or cx >= m.shape[1]:
        return {s: None for s in SIDES}
    keep = int(lab[cy, cx])
    if keep == 0:
        return {s: None for s in SIDES}
    return _sides_of((lab == keep).astype(np.uint8))


def profiles_of(img: np.ndarray, box, hole: bool = False, pad: int = 24) -> dict:
    """{'side': (outward_px, inward_px)} -- the continuous reading behind the three labels.

    Three labels (tab / notch / flat) throw away most of what the silhouette says, and one wrong
    label on one side is enough to break an exact pairing between five holes and five pieces.  The
    distances stay continuous, so a noisy side costs a little instead of everything.
    """
    x, y, w, h = box
    crop = vision.sub(img, (x - pad, y - pad, w + 2 * pad, h + 2 * pad))
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return {}
    if hole:
        m = read_hole_mask(img, box, pad)
    else:
        m = white_mask(img, box)
    if m.shape[0] < 20:
        return {}
    if int(m.sum()) < 200:
        return {}
    ys, xs = np.nonzero(m)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hh, ww = m.shape
    out: dict[str, tuple[float, float]] = {}
    for side in SIDES:
        if side == "top":
            prof = np.argmax(m, axis=0).astype(np.float32)
        elif side == "bottom":
            prof = (hh - 1 - np.argmax(m[::-1], axis=0)).astype(np.float32)
        elif side == "left":
            prof = np.argmax(m, axis=1).astype(np.float32)
        else:
            prof = (ww - 1 - np.argmax(m[:, ::-1], axis=1)).astype(np.float32)
        n = len(prof)
        mid = prof[n // 3: 2 * n // 3]
        core = float(np.median(prof))
        if side in ("top", "left"):
            outward, inward = core - float(mid.min()), float(mid.max()) - core
        else:
            outward, inward = float(mid.max()) - core, core - float(mid.min())
        out[side] = (outward, inward)
    return out


def read_hole_mask(img: np.ndarray, cell_box, pad: int = 24) -> np.ndarray:
    """The dark blob of an empty cell (its own shape), or an empty mask."""
    x, y, w, h = cell_box
    crop = vision.sub(img, (x - pad, y - pad, w + 2 * pad, h + 2 * pad))
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return np.zeros((1, 1), np.uint8)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    m = (g < HOLE_DARK).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    cy, cx = pad + h // 2, pad + w // 2
    if n <= 1 or cy >= m.shape[0] or cx >= m.shape[1]:
        return np.zeros((1, 1), np.uint8)
    keep = int(lab[cy, cx])
    if keep == 0:
        return np.zeros((1, 1), np.uint8)
    return (lab == keep).astype(np.uint8)
