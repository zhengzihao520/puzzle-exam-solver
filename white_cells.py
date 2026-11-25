"""Work out what an empty cell *requires* on each side, without touching the mouse.

The white cells are the reason this exists.  Their panel tiles are plain white (mean 255, no
drawing) and an empty board cell renders no shape at all (measured: flat board colour 23.0 at every
blank cell, same as the gaps between cells), so a white cell cannot say which white piece belongs in
it.  Its *neighbours* can:

  * a side on the board boundary must be flat,
  * an internal side must be the complement of what the neighbour shows on the shared edge.

Reading that edge is the interesting part.  Inside the board two placed pieces meet bright-on-bright,
so nothing can be thresholded -- measured on a real board, a silhouette reader called every internal
side "flat" (24 impossible both-flat pairs, 0 valid tab/notch pairs).  But an *empty* cell is dark
(23.0) and its neighbour is a photo, so the shared edge is bright-versus-dark, tabs and all.

So each side is measured as one signed number, `reach`: how far the neighbour's material goes past
the nominal cell edge, positive meaning it intrudes into our cell (we need a notch there) and
negative meaning it falls short (we need a tab).  The middle third of the side is what decides, which
is where the engine puts the cut.
"""
from __future__ import annotations

import cv2
import numpy as np

import vision

SIDES = ("top", "right", "bottom", "left")
OPP = {"tab": "notch", "notch": "tab", "flat": "flat", None: None}
BRIGHT = 45       # board background measures 23.0; any piece is far above this
CUT_PX = 9        # half a tab (the engine's tab is ~19 px): a safe "this is a cut" threshold
OUT = 26          # how far past the cell edge to look
IN = 12           # how far inside our own cell to look


def _strip(img: np.ndarray, box, side: str):
    """Crop across the shared edge, plus the index in it that is the nominal cell edge."""
    x, y, w, h = box
    if side == "top":
        return vision.sub(img, (x, y - IN, w, IN + OUT)), IN
    if side == "bottom":
        return vision.sub(img, (x, y + h - OUT, w, OUT + IN)), OUT
    if side == "left":
        return vision.sub(img, (x - IN, y, IN + OUT, h)), IN
    return vision.sub(img, (x + w - OUT, y, OUT + IN, h)), OUT


def _reach(img: np.ndarray, box, side: str) -> float | None:
    """Signed intrusion of the neighbour's material past the shared edge, in pixels.

    +  means it comes into our cell, so our piece needs a notch there.
    -  means it stops short, so our piece needs a tab.
    0  means it ends on the edge (only legal on the board boundary).
    """
    crop, edge = _strip(img, box, side)
    if crop.size == 0:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bright = g > BRIGHT

    if side in ("top", "bottom"):
        band = bright[:, crop.shape[1] // 3: 2 * crop.shape[1] // 3]
        if not band.any():
            return None
        if side == "top":                      # material lives above the edge
            return float(np.mean([np.nonzero(band[:, i])[0].max() - edge
                                  for i in range(band.shape[1]) if band[:, i].any()]))
        return float(np.mean([edge - np.nonzero(band[:, i])[0].min()
                              for i in range(band.shape[1]) if band[:, i].any()]))
    band = bright[crop.shape[0] // 3: 2 * crop.shape[0] // 3, :]
    if not band.any():
        return None
    if side == "left":
        return float(np.mean([np.nonzero(band[i, :])[0].max() - edge
                              for i in range(band.shape[0]) if band[i, :].any()]))
    return float(np.mean([edge - np.nonzero(band[i, :])[0].min()
                          for i in range(band.shape[0]) if band[i, :].any()]))


def required(img: np.ndarray, cells: dict, key: str, n: int, empty: set[str]) -> dict:
    """{'top': 'flat'|'tab'|'notch'|None, ...} for one empty cell."""
    r, c = (int(v) for v in key.split(","))
    box = cells[key]
    out: dict[str, str | None] = {}
    for side in SIDES:
        if {"top": r == 0, "bottom": r == n - 1, "left": c == 0, "right": c == n - 1}[side]:
            out[side] = "flat"
            continue
        nb = {"top": f"{r - 1},{c}", "bottom": f"{r + 1},{c}",
              "left": f"{r},{c - 1}", "right": f"{r},{c + 1}"}[side]
        if nb in empty:
            out[side] = None                   # hole next door: nothing to read
            continue
        d = _reach(img, box, side)
        if d is None:
            out[side] = None
        elif d > CUT_PX:
            out[side] = "notch"                # neighbour protrudes into us
        elif d < -CUT_PX:
            out[side] = "tab"                  # neighbour is indented, we must reach in
        else:
            out[side] = None
    return out


def agreement(sig: dict, need: dict) -> int:
    """Sides a candidate satisfies: +1 match, -1 mismatch, 0 when either side is unknown."""
    score = 0
    for side, want in need.items():
        if want is None:
            continue
        have = sig.get(side)
        if have is None:
            continue
        score += 1 if have == want else -1
    return score


def consider(sig: dict, need: dict) -> bool:
    """No known side of the requirement is contradicted by the piece."""
    return agreement(sig, need) >= 0
