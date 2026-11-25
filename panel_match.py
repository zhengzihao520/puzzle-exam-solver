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


def _prep(img: np.ndarray, box, extra: int = 0) -> np.ndarray:
    """Crop, then resample so that a piece and a panel cell end up at the same scale.

    A tray piece is about 112x85 px and a panel cell about 90x68, and they show the same picture,
    so both are normalised to SIZE.  `extra` widens the crop by that many of the cell's own pixels
    and widens the target size by the same proportion, which leaves room to slide the piece around
    inside the cell without changing the scale it is compared at.
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
    den = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    return float((a * b).sum() / den) if den else 0.0


EXTRA = 9          # room to slide the piece inside the cell, in the panel cell's own pixels


def similarity(piece: np.ndarray, cell_padded: np.ndarray) -> float:
    """Best normalised correlation of the piece anywhere inside the padded cell.

    matchTemplate does the whole shift search in one call and, unlike the hand-written loop it
    replaces, works on colour -- and colour is what separates one class member from another: a
    corgi and a beagle differ in hue, not in brightness.  The search range is `EXTRA` pixels of
    the panel cell, which covers the misalignment between a tray piece and a panel tile.
    """
    if cell_padded.shape[0] < piece.shape[0] or cell_padded.shape[1] < piece.shape[1]:
        return -1.0
    return float(cv2.matchTemplate(cell_padded, piece, cv2.TM_CCOEFF_NORMED).max())

def _scores(img: np.ndarray, pieces, cells: dict[str, tuple]) -> dict[tuple[str, str], float]:
    panel = {k: _prep(img, b, EXTRA) for k, b in cells.items()}
    scores: dict[tuple[str, str], float] = {}
    for pk, box in pieces.box.items():
        if pk in getattr(pieces, "used", ()):      # already placed this run
            continue
        mine = _prep(img, box)
        for ck, crop in panel.items():
            scores[(pk, ck)] = similarity(mine, crop)
    return scores


def plan_from_panel(img: np.ndarray, pieces, cells: dict[str, tuple],
                    floor: float = -1.0) -> tuple[dict[str, str], dict[str, float]]:
    """({cell key: piece key}, {cell: score}) from picture matching alone.

    Greedy on the strongest remaining pair: with the own cell winning 20 times out of 21 the
    ranking is sharp, and greedy keeps the code readable.  `cells` should be the cells that still
    need a piece -- a filled cell's picture matches a placed piece and would steal its score.
    """
    plan, conf, _margin = plan_and_margins(img, pieces, cells, floor)
    return plan, conf


def plan_and_margins(img: np.ndarray, pieces, cells: dict[str, tuple],
                     floor: float = -1.0) -> tuple[dict[str, str], dict[str, float],
                                                    dict[str, float]]:
    """As plan_from_panel, plus how far ahead the winning piece is for each cell.

    The margin is the tool against a real failure: the engine snaps on *geometry* alone, so a
    piece can lock into a different cell that happens to have the same cuts, and once locked it
    cannot be lifted -- its own cell is then left with nothing that fits.  Measured on three
    Part 3 runs, that is how the last cell was lost.  A cell whose picture match is far ahead of
    every other piece is safe to fill early; a cell where two pieces are close is exactly the one
    that can steal, so those are the ones to leave until the tray has thinned out.
    """
    scores = _scores(img, pieces, cells)
    used_pieces: set[str] = set()
    plan: dict[str, str] = {}
    conf: dict[str, float] = {}
    for (pk, ck), sc in sorted(scores.items(), key=lambda kv: -kv[1]):
        if pk in used_pieces or ck in plan or sc < floor:
            continue
        plan[ck] = pk
        conf[ck] = sc
        used_pieces.add(pk)

    margin: dict[str, float] = {}
    for ck in cells:
        ranked = sorted((sc for (pk, c2), sc in scores.items() if c2 == ck), reverse=True)
        margin[ck] = ranked[0] - ranked[1] if len(ranked) > 1 else 1.0
    return plan, conf, margin
