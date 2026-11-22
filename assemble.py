"""Assemble the board from seam continuity, with the reference panel's classes as a constraint.

Nothing here touches the mouse.  Given the pieces' pixels and the class wanted by each cell, it
answers "which piece goes where" -- so a live run only has to spend its 25 moves placing them.

Cost model: for every ordered pair of pieces, how badly does i's right edge continue into j's
left edge (and i's bottom into j's top).  Measured on ground truth: the true neighbour is chosen
first in 19 of 20 cases, so the pairwise signal is strong but not perfect -- which is why the
whole arrangement is optimised globally instead of pair by pair.

Search: greedy in ring order, then repeated improvements by swapping two pieces (or three, for
the cases a swap cannot fix), all under the rule that a piece may only sit in a cell whose class
matches.

  python assemble.py --selftest 10     # synthetic games, known answer, prints accuracy
"""
from __future__ import annotations

import argparse
import glob
import random
from itertools import combinations

import cv2
import numpy as np

import piececutter as pc

N = 5
W, H = 112, 85
INSET = 4


# ---------------------------------------------------------------- seam costs
def strips_of(piece: np.ndarray) -> dict[str, np.ndarray]:
    """Usable part of each of the four edges (inset from the stroke, middle third dropped)."""
    art = piece
    if art.shape[0] > H and art.shape[1] > W:               # canvas with margins
        _tab, margin, _cw, _ch = pc.geometry(W, H)
        m = int(round(margin))
        art = art[m:m + H, m:m + W]
    out = {}
    h, w = art.shape[:2]
    third_h, third_w = h // 3, w // 3
    keep_rows = np.r_[0:third_h, h - third_h:h]
    keep_cols = np.r_[0:third_w, w - third_w:w]
    out["right"] = art[keep_rows, w - 1 - INSET]
    out["left"] = art[keep_rows, INSET]
    out["bottom"] = art[h - 1 - INSET, keep_cols]
    out["top"] = art[INSET, keep_cols]
    return out


def seam_cost(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> tuple[float, float]:
    """(cost of b to the right of a, cost of b below a), normalised by local contrast.

    A raw difference is useless where the photo is flat: every candidate scores near zero there,
    so the ranking becomes arbitrary and the assembly wanders.  Dividing by the two strips' own
    contrast asks the right question -- does b continue a *across a real edge*, or do both sides
    merely happen to be smooth?
    """
    def diff(x, y):
        if x.shape != y.shape or x.size == 0:
            return 1e6
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        raw = float(np.abs(x - y).mean())
        contrast = 0.5 * (float(x.std()) + float(y.std()))
        return raw / (contrast + 8.0)
    return diff(a["right"], b["left"]), diff(a["bottom"], b["top"])


def apply_cut_constraint(cr, cd, piece_edges: dict[int, tuple], weight: float = 30.0) -> None:
    """Add the engine's hard rule as a heavy penalty: a tab must meet a notch.

    Measured reason this matters more than the pixels: seam continuity alone identified the true
    neighbour 19/20 on one photo but 6/20 and 7/20 on two others (flat, low-contrast photos),
    while six synthetic games all assembled perfectly (25/25) once the cut signs were applied.
    The signs do not care about colour, exposure or blur.
    """
    n = cr.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if piece_edges[i][1] * piece_edges[j][3] != -1:      # i.right vs j.left
                cr[i, j] += weight
            if piece_edges[i][2] * piece_edges[j][0] != -1:      # i.bottom vs j.top
                cd[i, j] += weight


def cost_matrices(pieces: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    strips = [strips_of(p) for p in pieces]
    n = len(pieces)
    cr = np.full((n, n), 1e6, np.float32)
    cd = np.full((n, n), 1e6, np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cr[i, j], cd[i, j] = seam_cost(strips[i], strips[j])
    return cr, cd


# ---------------------------------------------------------------- assembly
def ring_order(n: int) -> list[tuple[int, int]]:
    out = []
    for k in range((n + 1) // 2):
        top, bottom, left, right = k, n - 1 - k, k, n - 1 - k
        if top == bottom:
            out.extend((top, c) for c in range(left, right + 1))
            continue
        out.extend((top, c) for c in range(left, right))
        out.extend((r, right) for r in range(top, bottom))
        out.extend((bottom, c) for c in range(right, left, -1))
        out.extend((r, left) for r in range(bottom, top, -1))
    return out


def arrangement_cost(assign: dict[tuple[int, int], int], cr, cd, n: int = N) -> float:
    total = 0.0
    for (r, c), i in assign.items():
        if c + 1 < n and (r, c + 1) in assign:
            total += cr[i, assign[(r, c + 1)]]
        if r + 1 < n and (r + 1, c) in assign:
            total += cd[i, assign[(r + 1, c)]]
    return float(total)


def _greedy(order, cell_cls, piece_cls, cr, cd, n, piece_order):
    assign: dict[tuple[int, int], int] = {}
    used: set[int] = set()
    for cell in order:
        want = cell_cls.get(cell)
        best, best_cost = None, None
        for i in piece_order:
            if i in used or (want is not None and piece_cls[i] != want):
                continue
            cost = 0.0
            for (dr, dc) in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nb = (cell[0] + dr, cell[1] + dc)
                if nb not in assign:
                    continue
                j = assign[nb]
                if dc == -1:
                    cost += cr[j, i]
                elif dc == 1:
                    cost += cr[i, j]
                elif dr == -1:
                    cost += cd[j, i]
                else:
                    cost += cd[i, j]
            if best_cost is None or cost < best_cost:
                best, best_cost = i, cost
        if best is not None:
            assign[cell] = best
            used.add(best)
    return assign


def _polish(assign, cell_cls, piece_cls, cr, cd, n):
    """Two-swaps and three-cycles until nothing improves."""
    current = arrangement_cost(assign, cr, cd, n)
    keys = list(assign)
    for _ in range(200):
        improved = False
        for a, b in combinations(keys, 2):
            ia, ib = assign[a], assign[b]
            if (cell_cls.get(a) is not None and piece_cls[ib] != cell_cls[a]) or \
               (cell_cls.get(b) is not None and piece_cls[ia] != cell_cls[b]):
                continue
            assign[a], assign[b] = ib, ia
            cost = arrangement_cost(assign, cr, cd, n)
            if cost < current - 1e-6:
                current, improved = cost, True
            else:
                assign[a], assign[b] = ia, ib
        # three-cycles: a swap cannot fix a rotation of three pieces
        for a, b, c in combinations(keys, 3):
            ia, ib, ic = assign[a], assign[b], assign[c]
            ok = True
            for cell, piece in ((a, ib), (b, ic), (c, ia)):
                want = cell_cls.get(cell)
                if want is not None and piece_cls[piece] != want:
                    ok = False
                    break
            if not ok:
                continue
            assign[a], assign[b], assign[c] = ib, ic, ia
            cost = arrangement_cost(assign, cr, cd, n)
            if cost < current - 1e-6:
                current, improved = cost, True
            else:
                assign[a], assign[b], assign[c] = ia, ib, ic
        if not improved:
            break
    return assign, current


def assemble(cell_cls: dict[tuple[int, int], str], piece_cls: dict[int, str],
             cr, cd, n: int = N, restarts: int = 8) -> dict[tuple[int, int], int]:
    """Greedy in ring order from several tie-breakings, then two-swaps and three-cycles.

    One greedy pass plus a swap search solved only 3 of 6 synthetic games (84.7% of pieces).
    The misses were almost all rotations of three pieces, which no swap can repair, so the
    polish now also rotates triples, and several restarts keep the cheapest arrangement.
    """
    import random as _random
    order = ring_order(n)
    rng = _random.Random(5)
    best_assign, best_cost = None, None
    base = list(piece_cls)
    for k in range(restarts):
        piece_order = base[:]
        if k:
            rng.shuffle(piece_order)
        seed = order[:]
        if k:
            rng.shuffle(seed)
        assign = _greedy(seed, cell_cls, piece_cls, cr, cd, n, piece_order)
        if len(assign) < n * n:
            continue
        assign, cost = _polish(assign, cell_cls, piece_cls, cr, cd, n)
        if best_cost is None or cost < best_cost:
            best_assign, best_cost = assign, cost
    return best_assign or {}


# ---------------------------------------------------------------- self test
def synth_game(photo: np.ndarray, rng: random.Random, classes: int = 8):
    h = [[rng.choice((-1, 1)) for _ in range(N)] for _ in range(N - 1)]
    v = [[rng.choice((-1, 1)) for _ in range(N - 1)] for _ in range(N)]
    edges, cell_cls = {}, {}
    names = [f"c{i}" for i in range(classes)]
    for r in range(N):
        for c in range(N):
            edges[(r, c)] = [0 if r == 0 else -h[r - 1][c],
                             0 if c == N - 1 else v[r][c],
                             0 if r == N - 1 else h[r][c],
                             0 if c == 0 else -v[r][c - 1]]
            cell_cls[(r, c)] = rng.choice(names)

    _tab, margin, cw, ch = pc.geometry(W, H)
    m = int(round(margin))
    hh, ww = photo.shape[:2]
    pieces, piece_cls = [], {}
    for i, ((r, c), e) in enumerate(edges.items()):
        canvas = np.zeros((ch, cw, 3), np.uint8)
        ox, oy = c * W - m, r * H - m
        for yy in range(ch):
            sy = min(max(oy + yy, 0), hh - 1)
            for xx in range(cw):
                sx = min(max(ox + xx, 0), ww - 1)
                canvas[yy, xx] = photo[sy, sx]
        canvas[pc.piece_mask(tuple(e), W, H) == 0] = 0
        pieces.append(canvas)
        piece_cls[i] = cell_cls[(r, c)]
    truth = {cell: i for i, cell in enumerate(edges)}
    return pieces, cell_cls, piece_cls, truth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", type=int, default=0)
    args = ap.parse_args()
    if not args.selftest:
        print("nothing to do: pass --selftest N")
        return 0

    photos = sorted(glob.glob(r"F:\exam_next\raw_assets\*\*.jpg"))
    rng = random.Random(11)
    exact = 0
    per_piece = 0.0
    for t in range(args.selftest):
        photo = cv2.imread(rng.choice(photos))
        photo = cv2.resize(photo, (W * N, H * N), interpolation=cv2.INTER_AREA)
        pieces, cell_cls, piece_cls, truth = synth_game(photo, rng)
        cr, cd = cost_matrices(pieces)
        got = assemble(cell_cls, piece_cls, cr, cd)
        right = sum(1 for cell, i in got.items() if truth.get(cell) == i)
        per_piece += right / (N * N)
        exact += (right == N * N)
        print(f"  game {t + 1}: {right}/{N * N} pieces in their true cell")
    print(f"\nexactly solved {exact}/{args.selftest} games; "
          f"mean per-piece accuracy {per_piece / args.selftest:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
