"""Assemble the board from seam continuity, with the reference panel's classes as a constraint.

Nothing here touches the mouse.  Given the pieces' pixels and the class wanted by each cell, it
answers "which piece goes where" -- so a live run only has to spend its 25 moves placing them.

Cost model: for every ordered pair of pieces, how badly does i's right edge continue into j's
left edge (and i's bottom into j's top).  Measured on random photos, this cue is *not*
reliably strong on its own: the true neighbour came first in 19/20 pairs on one favourable
photo, but 6/20 and 7/20 on two flat, low-contrast ones -- and six synthetic games then came
out at 3/6 exact (86.7% of pieces).  So seam continuity is used as the tie-breaker it is good
at, and the cut signs (tab must meet notch, colour-blind) carry the arrangement: 6/6 exact.

Weight matters more than it looks.  At weight 30 a wrong sign outranks every pixel cost, so
errors stop being nudges and become commands -- at 20% wrong signs the result fell to 83.3%,
*worse* than ignoring the signs.  At weight 3 it is still 6/6 when clean and 98.7% at 10%.

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


def apply_cut_constraint(cr, cd, piece_edges: dict[int, tuple], weight: float = 3.0) -> None:
    """Add the engine's rule as a penalty: a tab must meet a notch.

    Measured reason this matters more than the pixels: seam continuity alone identified the true
    neighbour 19/20 on one photo but 6/20 and 7/20 on two others (flat, low-contrast photos),
    while six synthetic games all assembled perfectly (25/25) once the cut signs were applied.
    The signs do not care about colour, exposure or blur.
    A side that could not be read (None) is not evidence of anything and is skipped -- charging
    it as a mismatch would punish every pair involving one unread edge equally and quietly turn
    the reader's blind spots into instructions.  A *flat* side (0) is the opposite: it is real
    evidence, because only a border piece has one, so the product test below keeps it out of
    every interior position.
    """
    n = cr.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r_i, l_j = piece_edges[i][1], piece_edges[j][3]       # i.right vs j.left
            if r_i is not None and l_j is not None and r_i * l_j != -1:
                cr[i, j] += weight
            b_i, t_j = piece_edges[i][2], piece_edges[j][0]       # i.bottom vs j.top
            if b_i is not None and t_j is not None and b_i * t_j != -1:
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


def _fits(piece_cls: dict, i: int, want) -> bool:
    """May piece i sit in a cell that wants `want`?

    A piece's class is often a two-way vote: the catalog and the classifier disagree, and the
    reading is then "one of these", not "this".  Collapsing that to one label is what left two
    white pieces unable to reach the five white cells they belonged to.  So a set means "any of
    these", None means "not sure enough to rule anything out", a plain string stays exact.
    """
    v = piece_cls.get(i)
    if want is None or v is None:
        return True
    return want in v if isinstance(v, (set, frozenset)) else want == v


def _greedy(order, cell_cls, piece_cls, cr, cd, n, piece_order):
    assign: dict[tuple[int, int], int] = {}
    used: set[int] = set()
    for cell in order:
        want = cell_cls.get(cell)
        best, best_cost = None, None
        for i in piece_order:
            if i in used or not _fits(piece_cls, i, want):
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
            if not _fits(piece_cls, ib, cell_cls.get(a)) or \
               not _fits(piece_cls, ia, cell_cls.get(b)):
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
                if not _fits(piece_cls, piece, want):
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

    Incomplete arrangements are returned rather than discarded.  Demanding all n*n cells meant
    that one cell nobody could legally fill -- a cell wanting umbrella on a tray holding no
    umbrella, after the reader had already shortened five white pieces to three -- threw away
    24 perfectly good placements with it.  More filled cells wins over a cheaper partial one.
    """
    import random as _random
    # Only cells the caller actually named: on a half-finished board that is a handful, and
    # filling the rest of the 5x5 with them produced more placements than there were cells.
    order = [c for c in ring_order(n) if c in cell_cls]
    rng = _random.Random(5)
    best_assign, best_key = None, None
    base = list(piece_cls)
    for k in range(restarts):
        piece_order = base[:]
        if k:
            rng.shuffle(piece_order)
        seed = order[:]
        if k:
            rng.shuffle(seed)
        assign = _greedy(seed, cell_cls, piece_cls, cr, cd, n, piece_order)
        if not assign:
            continue
        if len(assign) == len(order):
            assign, cost = _polish(assign, cell_cls, piece_cls, cr, cd, n)
        else:
            cost = arrangement_cost(assign, cr, cd, n)
        key = (len(assign), -cost)
        if best_key is None or key > best_key:
            best_assign, best_key = assign, key
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
    pieces, piece_cls, piece_edges = [], {}, {}
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
        piece_edges[i] = tuple(e)
    truth = {cell: i for i, cell in enumerate(edges)}
    return pieces, cell_cls, piece_cls, piece_edges, truth


def _corrupt(piece_edges: dict[int, tuple], p: float, rng: random.Random):
    """Simulate the tray reader: each side may come back unread (None) or simply wrong."""
    out = {}
    for i, e in piece_edges.items():
        s = list(e)
        for k in range(4):
            if rng.random() < p:
                s[k] = None if rng.random() < 0.5 else -s[k]
        out[i] = tuple(s)
    return out


def _score(assign: dict, truth: dict) -> int:
    return sum(1 for cell, i in assign.items() if truth.get(cell) == i)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", type=int, default=0)
    ap.add_argument("--err", type=float, default=0.0,
                    help="fraction of cut sides the reader gets wrong (0 = perfect)")
    ap.add_argument("--weight", type=float, default=3.0,
                    help="how hard a mismatched tab/notch pushes the cost; 30 makes wrong "
                         "signs dominate (20%% errors -> 83%%, worse than pixels alone), "
                         "3 costs nothing when the signs are clean and degrades gently")
    args = ap.parse_args()
    if not args.selftest:
        print("nothing to do: pass --selftest N")
        return 0

    photos = sorted(glob.glob(r"F:\exam_next\raw_assets\*\*.jpg"))
    rng = random.Random(11)
    exact = [0, 0]
    per_piece = [0.0, 0.0]
    for t in range(args.selftest):
        photo = cv2.imread(rng.choice(photos))
        photo = cv2.resize(photo, (W * N, H * N), interpolation=cv2.INTER_AREA)
        pieces, cell_cls, piece_cls, piece_edges, truth = synth_game(photo, rng)
        cr, cd = cost_matrices(pieces)

        read = _corrupt(piece_edges, args.err, rng) if args.err else piece_edges
        cr2, cd2 = cr.copy(), cd.copy()
        apply_cut_constraint(cr2, cd2, read, weight=args.weight)

        plain = assemble(cell_cls, piece_cls, cr, cd)
        cut = assemble(cell_cls, piece_cls, cr2, cd2)
        r0, r1 = _score(plain, truth), _score(cut, truth)
        per_piece[0] += r0 / (N * N)
        per_piece[1] += r1 / (N * N)
        exact[0] += (r0 == N * N)
        exact[1] += (r1 == N * N)
        print(f"  game {t + 1}: pixels only {r0}/{N * N}   with cut signs {r1}/{N * N}")

    for k, name in enumerate(("pixels only", "with cut signs")):
        print(f"{name:15s} exactly solved {exact[k]}/{args.selftest} games, "
              f"mean per-piece accuracy {per_piece[k] / args.selftest:.1%}")
    if args.err:
        print(f"(cut signs corrupted on {args.err:.0%} of sides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
