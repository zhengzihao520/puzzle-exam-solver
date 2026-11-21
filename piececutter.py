"""Exact jigsaw silhouette generator, ported from the site's own engine.

`engine_PieceCutter.js` (archived in recon/, upstream playjigsaw/play-jigsaw-puzzles-online)
draws each piece with four Bezier edges and clips the photo to that path, so the silhouette
depends *only* on the four edge types -- not on the photo.  That means the shape of a piece
can be recovered by rendering all 81 combinations (each side flat / tab / notch) and picking
the one that best overlaps the observed silhouette, which is far steadier than thresholding
protrusion profiles.

Geometry, straight from the engine:
    tabSize = min(pieceW, pieceH) * 0.22
    margin  = tabSize * 1.5
    canvas  = (pieceW + 2*margin) x (pieceH + 2*margin)
"""
from __future__ import annotations

import itertools

import cv2
import numpy as np

FLAT, TAB, NOTCH = 0, 1, -1
SIDE_NAMES = ("top", "right", "bottom", "left")
KEY_TO_VALUE = {"flat": FLAT, "tab": TAB, "notch": NOTCH}
VALUE_TO_KEY = {v: k for k, v in KEY_TO_VALUE.items()}


def geometry(piece_w: float, piece_h: float) -> tuple[float, float, int, int]:
    tab = min(piece_w, piece_h) * 0.22
    margin = tab * 1.5
    return tab, margin, int(round(piece_w + margin * 2)), int(round(piece_h + margin * 2))


def _cubic(p0, p1, p2, p3, steps: int = 12) -> list[tuple[float, float]]:
    out = []
    for i in range(1, steps + 1):
        t = i / steps
        u = 1 - t
        out.append((u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                    u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]))
    return out


def edge_points(x1: float, y1: float, x2: float, y2: float, kind: int, tab: float
                ) -> list[tuple[float, float]]:
    """Port of PieceCutter.drawEdge: the same seven control points, sampled."""
    if kind == FLAT:
        return [(x2, y2)]
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length == 0:
        return [(x2, y2)]
    nx, ny = -dy / length, dx / length
    sign = kind
    p1 = (x1 + dx * 0.38, y1 + dy * 0.38)
    p2 = (x1 + dx * 0.42 + nx * tab * 0.3 * sign, y1 + dy * 0.42 + ny * tab * 0.3 * sign)
    p3 = (x1 + dx * 0.35 + nx * tab * 1.05 * sign, y1 + dy * 0.35 + ny * tab * 1.05 * sign)
    p4 = (x1 + dx * 0.5 + nx * tab * 1.15 * sign, y1 + dy * 0.5 + ny * tab * 1.15 * sign)
    p5 = (x1 + dx * 0.65 + nx * tab * 1.05 * sign, y1 + dy * 0.65 + ny * tab * 1.05 * sign)
    p6 = (x1 + dx * 0.58 + nx * tab * 0.3 * sign, y1 + dy * 0.58 + ny * tab * 0.3 * sign)
    p7 = (x1 + dx * 0.62, y1 + dy * 0.62)
    pts = [p1]
    pts += _cubic(p1, p2, p3, p4)
    pts += _cubic(p4, p5, p6, p7)
    pts += [(x2, y2)]
    return pts


def piece_mask(edges: tuple[int, int, int, int], piece_w: float = 106.0, piece_h: float = 83.6,
               canvas: tuple[int, int] | None = None) -> np.ndarray:
    """Binary silhouette of one piece; edges are (top, right, bottom, left)."""
    tab, margin, cw, ch = geometry(piece_w, piece_h)
    if canvas is not None:
        cw, ch = canvas
    ox = oy = margin
    x2, y2 = ox + piece_w, oy + piece_h
    path: list[tuple[float, float]] = [(ox, oy)]
    path += edge_points(ox, oy, x2, oy, edges[0], tab)
    path += edge_points(x2, oy, x2, y2, edges[1], tab)
    path += edge_points(x2, y2, ox, y2, edges[2], tab)
    path += edge_points(ox, y2, ox, oy, edges[3], tab)
    pts = np.array([[int(round(x)), int(round(y))] for x, y in path], dtype=np.int32)
    mask = np.zeros((ch, cw), np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def all_templates(piece_w: float = 106.0, piece_h: float = 83.6) -> list[tuple[tuple[int, int, int, int],
                                                                            np.ndarray]]:
    return [(e, piece_mask(e, piece_w, piece_h))
            for e in itertools.product((FLAT, TAB, NOTCH), repeat=4)]


def tidy(mask: np.ndarray, size: tuple[int, int] = (150, 130)) -> np.ndarray:
    """Crop to the piece bbox and rescale, so comparison ignores placement and scale."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros(size[::-1], np.uint8)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(crop, size, interpolation=cv2.INTER_NEAREST)


def classify(observed: np.ndarray, piece_w: float = 106.0, piece_h: float = 83.6,
             size: tuple[int, int] = (150, 130)) -> tuple[dict[str, str], float, list[tuple[float, str]]]:
    """Best edge combination for an observed silhouette.

    Returns (edges as names, best IoU, ranked list of the top few) so the caller can see how
    clear-cut the winner was.
    """
    obs = tidy(observed, size)
    scored = []
    for edges, tpl in all_templates(piece_w, piece_h):
        t = tidy(tpl, size)
        inter = np.logical_and(obs, t).sum()
        union = np.logical_or(obs, t).sum()
        iou = float(inter) / float(union) if union else 0.0
        name = "/".join(VALUE_TO_KEY[v] for v in edges)
        scored.append((iou, name))
    scored.sort(reverse=True)
    best_iou, best = scored[0]
    names = best.split("/")
    return dict(zip(SIDE_NAMES, names)), best_iou, scored[:3]


def flats_of(edges: dict[str, str]) -> list[str]:
    return sorted(s for s, k in edges.items() if k == "flat")
