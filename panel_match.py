"""Match each piece against the reference panel: the whole assignment, at no cost.

The shortest path to the problem, and one that was dismissed early on a wrong belief: that the panel
showed a *different* photo of the same class, which would have made picture matching useless.
Measured on a live board, the piece in a cell correlates 0.662 on average with its own panel cell
against 0.380 with the best wrong one, and the own cell wins in 20 of 21 cases.  The panel shows the
same picture.
"""
from __future__ import annotations

import cv2
import numpy as np

import vision

SIZE = (64, 48)
EXTRA = 9          # room to slide the piece inside the cell, in the panel cell's own pixels


def _prep(img: np.ndarray, box, extra: int = 0) -> np.ndarray:
    """Crop, then resample so a piece and a panel cell end up at the same scale.

    A tray piece is about 112x85 px and a panel cell about 90x68, and they show the same picture, so
    both are normalised to SIZE.  `extra` widens the crop by that many of the cell's own pixels and
    widens the target size by the same proportion, which leaves room to slide the piece around inside
    the cell without changing the scale it is compared at.
    """
    x, y, w, h = box
    crop = vision.sub(img, (x - extra, y - extra, w + 2 * extra, h + 2 * extra))
    tw = int(round(SIZE[0] * (1.0 + 2.0 * extra / w)))
    th = int(round(SIZE[1] * (1.0 + 2.0 * extra / h)))
    # colour, not grey: the photos differ mainly in what they show, not in how bright they are
    return cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation over all three channels at once."""
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den else -1.0


def similarity(piece: np.ndarray, cell_padded: np.ndarray) -> float:
    """Best normalised correlation of the piece anywhere inside the padded cell.

    matchTemplate does the whole shift search in one call and, unlike the hand-written loop it
    replaces, works on colour -- and colour is what separates one class member from another: a corgi
    and a beagle differ in hue, not in brightness.  The search range is `EXTRA` pixels of the panel
    cell, which covers the misalignment between a tray piece and a panel tile.
    """
    if cell_padded.shape[0] < piece.shape[0] or cell_padded.shape[1] < piece.shape[1]:
        return -1.0
    return float(cv2.matchTemplate(cell_padded, piece, cv2.TM_CCOEFF_NORMED).max())


def scores(img: np.ndarray, pieces, cells: dict[str, tuple]) -> dict[tuple[str, str], float]:
    """The whole piece x cell correlation matrix.  Every caller below is built on this."""
    panel = {k: _prep(img, b, EXTRA) for k, b in cells.items()}
    out: dict[tuple[str, str], float] = {}
    for pk, box in pieces.box.items():
        if pk in getattr(pieces, "used", ()):      # already placed this run
            continue
        mine = _prep(img, box)
        for ck, crop in panel.items():
            out[(pk, ck)] = similarity(mine, crop)
    return out


def plan_and_margins(scores: dict[tuple[str, str], float],
                     cells) -> tuple[dict[str, str], dict[str, float], dict[str, float]]:
    """(plan, confidence, margin) from a score matrix.

    Greedy on the strongest remaining pair: with the own cell winning 20 times out of 21 the ranking
    is sharp, and greedy keeps the code readable.

    `margin` is the gap between the winner and the runner-up for each cell.  A cell where two pieces
    are close is the one that can be stolen -- the engine snaps on geometry, so the wrong one of the
    two can lock into a same-shaped neighbour and never come out again -- which is why those cells are
    filled last.
    """
    used_pieces: set[str] = set()
    plan: dict[str, str] = {}
    conf: dict[str, float] = {}
    for (pk, ck), sc in sorted(scores.items(), key=lambda kv: -kv[1]):
        if pk in used_pieces or ck in plan:
            continue
        plan[ck] = pk
        conf[ck] = sc
        used_pieces.add(pk)

    margin: dict[str, float] = {}
    for ck in cells:
        ranked = sorted((sc for (pk, c2), sc in scores.items() if c2 == ck), reverse=True)
        margin[ck] = ranked[0] - ranked[1] if len(ranked) > 1 else 1.0
    return plan, conf, margin


def best_cell(scores: dict[tuple[str, str], float]) -> dict[str, str]:
    """{piece: the cell it looks most like}, independent of the one-to-one plan."""
    out: dict[str, str] = {}
    seen: dict[str, tuple[float, str]] = {}
    for (pk, ck), sc in scores.items():
        if pk not in seen or sc > seen[pk][0]:
            seen[pk] = (sc, ck)
    for pk, (sc, ck) in seen.items():
        out[pk] = ck
    return out


def plan_from_panel(img: np.ndarray, pieces, cells: dict[str, tuple],
                    floor: float = -1.0) -> tuple[dict[str, str], dict[str, float]]:
    """({cell key: piece key}, {cell: score}) from picture matching alone.

    Kept for callers that only need the plan; `cells` should be the cells that still need a piece,
    because a filled cell's picture matches a placed piece and would steal its score.
    """
    plan, conf, _margin = plan_and_margins(scores(img, pieces, cells), cells)
    return plan, conf
