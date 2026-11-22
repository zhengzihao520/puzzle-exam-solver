"""Recover a piece's exact body rectangle by fitting the engine's 81 silhouettes.

The tray slot is not the piece: a piece sits inside its slot with a few pixels of slack, its
outline is a jigsaw shape and a notch lets the background through.  Comparing a slot-shaped
crop against the catalog lands around 0.45, while the same comparison on the reference
panel's exact rectangles lands at 0.87 on the same frame -- so the gap is alignment, not the
catalog.

piececutter renders the engine's own Bezier geometry for all 81 edge combinations, so the
body rectangle is recovered by fitting: place each template at each offset and keep the best
overlap.  That yields the body rect *and* the four edge types.
"""
from __future__ import annotations

from itertools import product

import cv2
import numpy as np

import piececutter as pc

BODY_W, BODY_H = 112, 85


def silhouette(img: np.ndarray, box, pad: int = 34) -> tuple[np.ndarray, tuple[int, int]]:
    x, y, w, h = box
    X0, Y0 = max(0, x - pad), max(0, y - pad)
    win = img[Y0:y + h + pad, X0:x + w + pad]
    g = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)
    m = (g > 20).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return np.zeros_like(m), (X0, Y0)
    cy, cx = (y + h // 2) - Y0, (x + w // 2) - X0
    seed = lab[cy, cx] if 0 <= cy < lab.shape[0] and 0 <= cx < lab.shape[1] else 0
    if seed == 0:
        seed = 1 + int(np.argmax(st[1:, 4]))
    m = (lab == seed).astype(np.uint8)
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
    m = np.clip(m + (ff == 0).astype(np.uint8), 0, 1)
    return m, (X0, Y0)


_TEMPLATES: list[tuple[tuple[int, int, int, int], np.ndarray, int]] | None = None


def templates() -> list[tuple[tuple[int, int, int, int], np.ndarray, int]]:
    global _TEMPLATES
    if _TEMPLATES is None:
        out = []
        for edges in product((0, 1, -1), repeat=4):
            m = pc.piece_mask(edges, BODY_W, BODY_H)
            out.append((edges, m, int(m.sum())))
        _TEMPLATES = out
    return _TEMPLATES


def fit(img: np.ndarray, box, radius: int = 12, step: int = 4):
    """(iou, edges, body_rect) for one slot, or None."""
    mask, (X0, Y0) = silhouette(img, box)
    if mask.sum() < 3000:
        return None
    _tab, margin, _cw, _ch = pc.geometry(BODY_W, BODY_H)
    tpls = templates()
    th, tw = tpls[0][1].shape
    H, W = mask.shape
    best = (-1.0, None, None)
    for oy in range(-radius, radius + 1, step):
        for ox in range(-radius, radius + 1, step):
            ty, tx = oy + mask.shape[0] // 2 - th // 2, ox + mask.shape[1] // 2 - tw // 2
            if ty < 0 or tx < 0 or ty + th > H or tx + tw > W:
                continue
            sub = mask[ty:ty + th, tx:tx + tw]
            a = int(sub.sum())
            if a == 0:
                continue
            inter = np.array([int(np.bitwise_and(sub, t).sum()) for _e, t, _s in tpls],
                             dtype=np.float32)
            areas = np.array([s for _e, _t, s in tpls], dtype=np.float32)
            iou = inter / np.maximum(a + areas - inter, 1)
            i = int(np.argmax(iou))
            if iou[i] > best[0]:
                body_x = int(round(X0 + tx + margin))
                body_y = int(round(Y0 + ty + margin))
                best = (float(iou[i]), tpls[i][0], (body_x, body_y, BODY_W, BODY_H))
    return best if best[0] > 0 else None
