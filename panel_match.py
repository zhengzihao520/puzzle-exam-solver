"""Match each tray piece to the reference panel cell it belongs to, by picture.

This is the shortest path to the problem, and it was dismissed early on a wrong belief: that the
panel showed a *different* photo of the same class, which would have made picture matching useless.
Measured on a live board, the piece in a cell correlates 0.662 on average with its own panel cell
against 0.380 with the best wrong one, and the own cell wins in 20 of 21 cases.  The panel shows the
same picture.

So a piece's destination can be read straight off the panel: no cut reading (which costs a move per
piece), no class guessing, and no refusals from picking the wrong piece of a class.  The whole
assignment costs zero moves, because the tray and the panel are both already on screen.
"""
from __future__ import annotations

import cv2
import numpy as np

import vision

SIZE = (64, 48)


def _grey(img: np.ndarray, box) -> np.ndarray:
    g = cv2.cvtColor(vision.sub(img, box), cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / den) if den else 0.0


def similarity(piece: np.ndarray, cell: np.ndarray, pad: int = 3) -> float:
    """Best correlation over small shifts, so a pixel or two of crop error is forgiven."""
    best = -1.0
    h, w = piece.shape
    canvas = cv2.copyMakeBorder(cell, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    for dy in range(0, 2 * pad + 1):
        for dx in range(0, 2 * pad + 1):
            best = max(best, _ncc(piece, canvas[dy:dy + h, dx:dx + w]))
    return best


def plan_from_panel(img: np.ndarray, pieces, cells: dict[str, tuple],
                    floor: float = -1.0) -> tuple[dict[str, str], dict[str, float]]:
    """({cell key: piece key}, {cell: score}) from picture matching alone.

    Greedy on the strongest remaining pair: with the own cell winning 20 times out of 21 the
    ranking is sharp, and greedy keeps the code readable.  `cells` should be the cells that still
    need a piece -- a filled cell's picture matches a placed piece and would steal its score.
    """
    panel = {k: _grey(img, b) for k, b in cells.items()}
    scores: dict[tuple[str, str], float] = {}
    for pk, box in pieces.box.items():
        if pk in getattr(pieces, "used", ()):  # already placed this run
            continue
        mine = _grey(img, box)
        for ck, crop in panel.items():
            scores[(pk, ck)] = similarity(mine, crop)

    used_pieces: set[str] = set()
    plan: dict[str, str] = {}
    conf: dict[str, float] = {}
    for (pk, ck), sc in sorted(scores.items(), key=lambda kv: -kv[1]):
        if pk in used_pieces or ck in plan or sc < floor:
            continue
        plan[ck] = pk
        conf[ck] = sc
        used_pieces.add(pk)
    return plan, conf
