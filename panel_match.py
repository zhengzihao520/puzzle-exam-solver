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

# 96x72 is the tuned resolution: at 64x48 the median own-cell correlation was 0.60 against 0.67 at
# 96x72, and every point of that correlation shows up as a wrong first attempt (two moves each).
SIZE = (96, 72)
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


def _assign_max(rows: list[str], cols: list[str],
                w: dict[tuple[str, str], float]) -> dict[str, str]:
    """Optimal one-to-one assignment {col: row} maximising the total weight.

    Greedy-on-the-strongest-pair was here first, on the argument that the own cell wins 20 times
    in 21 so the ranking is sharp.  What greedy cannot do is the one thing that matters for a
    25-move game: when two pieces both look most like the same cell, greedy takes the stronger of
    them and the other cell is left to a second-best candidate it never checked against the
    others.  The assignment is solved exactly instead (Kahn-Munkres, 30x25 costs microseconds),
    which is verified against brute force in tools/assign_test.py.
    """
    n, m = len(rows), len(cols)
    if not n or not m:
        return {}
    flip = n > m                       # the solver wants rows <= cols
    r_list, c_list = (cols, rows) if flip else (rows, cols)
    cost = [[-w.get((r, c), 0.0) if not flip else -w.get((c, r), 0.0)
             for c in c_list] for r in r_list]
    best = _kahn_munkres(cost)
    out: dict[str, str] = {}
    for i, j in best.items():
        if flip:
            out[r_list[i]] = c_list[j]      # r_list is the cells
        else:
            out[c_list[j]] = r_list[i]
    return out


def _kahn_munkres(cost: list[list[float]]) -> dict[int, int]:
    """{row: col} minimising the total cost, rows <= cols (the O(n^2 m) augmenting-path form)."""
    n, m = len(cost), len(cost[0])
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while p[j0]:                                   # while this column is taken
            used[j0] = True
            i0 = p[j0]
            delta, j1 = INF, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return {p[j] - 1: j - 1 for j in range(1, m + 1) if p[j]}


def plan_and_margins(scores: dict[tuple[str, str], float],
                     cells) -> tuple[dict[str, str], dict[str, float], dict[str, float]]:
    """(plan, confidence, margin) from a score matrix.

    `plan` is the optimal one-to-one assignment of pieces to cells under the picture scores.

    `margin` is the gap between the winner and the runner-up for each cell.  A cell where two
    pieces are close is the one that can be stolen -- the engine snaps on geometry, so the wrong
    one of the two can lock into a same-shaped neighbour and never come out again -- which is why
    those cells are filled last.
    """
    pieces = sorted({pk for pk, _c in scores})
    cell_keys = list(cells)
    # Pieces with no score at all are left out rather than assigned on a fabricated zero: the
    # matrix is only built from what was actually measured.
    pieces = [pk for pk in pieces if any((pk, ck) in scores for ck in cell_keys)]
    plan = _assign_max(pieces, cell_keys, scores)
    conf = {ck: scores[(pk, ck)] for ck, pk in plan.items() if (pk, ck) in scores}

    margin: dict[str, float] = {}
    for ck in cell_keys:
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
